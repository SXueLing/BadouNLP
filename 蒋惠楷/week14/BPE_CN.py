import logging
import collections
import threading
import os
import numpy as np
from types import SimpleNamespace
from typing import List, TypeVar
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from alive_progress import alive_bar
import concurrent.futures
from itertools import islice


logger = logging.getLogger("default")

T = TypeVar('T')
class WithMutex:
    def __init__(self, obj: T):
        self.obj = obj
        self.mutex = threading.Lock()

class MutexMap:
    '''多线程操作字典'''
    def __init__(self, map):
        self.map = map
        self.mutex = threading.Lock()

    def __getitem__(self, key):
        with self.mutex:
            return self.map[key]

    def __setitem__(self, key, value):
        with self.mutex:
            self.map[key] = value

    def __contains__(self, key):
        with self.mutex:
            return key in self.map

    def __len__(self):
        with self.mutex:
            return len(self.map)

    def __iter__(self):
        with self.mutex:
            return iter(self.map)

    def incr(self, key, value=1):
        with self.mutex:
            self.map[key] += value

class ParalledTask:
    """实现了对并行线程的封装"""
    @staticmethod
    def create(task_name) -> 'ParalledTask':
        """创建并行任务"""
        instance = ParalledTask()
        instance.task_name = task_name # 任务名称(仅用于日志)
        instance.results = {}  # 存储所有工作线程的返回结果(key是worker_id，value是执行结果)
        return instance
    
    def set_nworker(self, nworker: int) -> 'ParalledTask':
        """设置并行协程数"""
        self.nworker = nworker
        return self # 允许方法链式调用
    
    def set_worker_func(self, worker_func) -> 'ParalledTask':
        """设置并行协程执行器"""
        self.worker_func = worker_func
        return self
    
    def set_worker_args(self, worker_args) -> 'ParalledTask':
        """设置协程执行器参数"""
        self.worker_args = worker_args
        return self
    
    def set_worker_arg_provider_func(self, worker_arg_provider_func):
        """设置参数提供函数"""
        self.worker_arg_provider_func = worker_arg_provider_func
        return self
    
    def set_reducer_func(self, reducer_func) -> 'ParalledTask':
        """设置并行任务执行结果合并器"""
        self.reducer_func = reducer_func
        return self

    def set_progress_goal(self, goal: int) -> 'ParalledTask':
        """设置进度条目标值"""
        self.progress_goal = goal
        return self
    
    def execute(self) -> 'ParalledTask':
        """执行并行任务"""
        logger.info(f'{self.task_name} start')
        # 管理多个上下文管理器，动态处理资源
        with ExitStack() as stack, \
                ThreadPoolExecutor(max_workers=self.nworker) as executor: # 并发任务执行
            ctxs = []
            if hasattr(self, 'progress_goal'):
                goal = self.progress_goal
                # 创建进度条
                bar = stack.enter_context(alive_bar(goal))
                # 使用互斥锁封装进度条
                bar_with_mutex = WithMutex(bar)

            for worker_id in range(self.nworker):
                if (hasattr(self, 'worker_arg_provider_func')):
                    worker_arg = self.worker_arg_provider_func(
                        worker_id, self.nworker
                    )
                else:
                    worker_arg = self.worker_args
                
                worker_ctx = {
                    **worker_arg,
                    'worker_id': worker_id,
                    'task': self,
                    'bar': bar_with_mutex,
                }
                ctxs.append(worker_ctx)
            # 提交任务到执行器
            futures = [executor.submit(self.worker_func, ctxs[i]) for i in range(self.nworker)]
            # 等待完成,并收集执行结果
            for future in concurrent.futures.as_completed(futures):
                worker_id = futures.index(future)
                self.results[worker_id] = future.result()
            # 根据执行器id排序,避免乱序
            self.results = {k: v for k, v in sorted(self.results.items(), key=lambda item: item[0])}
        logger.info(f'{self.task_name} done')
        return self
    
    def get_results(self):
        """获取并行任务执行结果"""
        # 如果有合并器，则进行合并，否则直接返回结果
        if(self.reducer_func is None):
            return self.results
        return self.reducer_func(self.results)

class Tokenizer:
    def __init__(self):
        self.user_dict = {}

    def add_dict(self, new_dict: dict) -> 'Tokenizer':
        """添加用户词典"""
        self._preproc(new_dict)
        return self

    def _preproc(self, new_dict: dict):
        # merge new_dict to user_dict
        for k, v in new_dict.items():
            if k in self.user_dict:
                self.user_dict[k] += v
            else:
                self.user_dict[k] = v

        self.user_dict_items = sorted(self.user_dict.items(),
                                key=lambda x: (-len(x[0]), x[1]))
        pass

    def tokenize(self, spaced_sentence: list) -> list:
        """对句子分词"""
        si = 0
        result = []
        while si < len(spaced_sentence):
            matched = False
            prevlen = 0
            to_match = ''
            for w, c in self.user_dict_items:
                if prevlen != len(w):
                    to_match = ''.join(spaced_sentence[si:si + len(w)])
                    prevlen = len(w)
                if w == to_match:
                    result.append(w)
                    si += len(w)
                    matched = True
                    break
            if not matched:
                result.append(spaced_sentence[si])
                si += 1

        return result

class BpeCn:
    def __init__(self, options):
        """
        options:
            - train_file: 训练文件。
            - vocab_file: 输出文件。
            - nworkers: 并行协程数
        """
        # 加载内部训练数据
        logger.info("Loading train data from {} ...".format(options.train_file))
        self._train_lines = open(options.train_file, 'r', encoding='utf-8').readlines()
        logger.info("Loading train data (total {} lines) done.".format(len(self._train_lines)))
        self.vocab_file = options.vocab_file
        self.nworker = options.nworker
        self._preproc() # 预处理阶段
    
    def partition(self, lines: np.ndarray, nworker: int) -> List[List[str]]:
        """将训练数据分割为 nworker 份。"""
        datasets = []
        for i in range(nworker):
            datasets.append([])
        for i in range(len(lines)):
            datasets[i % nworker].append(lines[i])
        return datasets

    def _preproc(self):
        '''进行数据的预处理'''
        logger.info("Preprocessing train data ...")
        train_lines = [line.strip().split() for line in self._train_lines]
        self._train_lines = None
        self._train_lines_np = np.array(train_lines, dtype=object)
        puncs_zh = ['。', '，', '？', '！', '；', '：', '、', '（', '）', '「',
                    '」', '“', '”', '‘', '’', '《', '》', '【', '】', '…', '—', '～', '　']
        puncs_en = ['.', ',', '?', '!', ';', ':',
                    '(', ')', '"', '"', '\'', '\'', '<', '>', '[', ']', '...', '~']
        puncs = [*puncs_zh, *puncs_en]

        def _replace_worker(ctx: dict):
            task = ctx.get('task')
            bar = ctx.get('bar')
            worker_id = ctx.get('worker_id')
            line_strs: List[List[str]] = ctx.get('datasets')[worker_id]
            for line in line_strs:
                for i in range(len(line)):
                    if line[i] in puncs:
                        line[i] = "#"
                with bar.mutex: # 加锁
                    bar.obj()   # 进度条更新操作
        
        # 去除训练数据中的重复项
        logger.info("--removing duplication.")
        self._train_lines_np = np.unique(self._train_lines_np)
        logger.info("--removing duplication done.")

        self.datasets = self.partition(self._train_lines_np, self.nworker)
        
        ParalledTask.create('-- removing punc')\
            .set_nworker(self.nworker)\
            .set_worker_func(_replace_worker)\
            .set_progress_goal(len(self._train_lines_np))\
            .set_worker_args({'datasets': self.datasets})\
            .execute()
        logger.info("Preprocessing train data done.")
    
    def train(self):
        "训练BPE模型"
        # 创建字典，键为词，如“电脑”，值为词频，如“100”
        self.vocab_map = MutexMap(collections.defaultdict(int))
        # 创建并行数据集
        ds = self.datasets
        # 词频统计总表。用于多个执行器间共享
        shared_freq_stat_map = None
        min_thold = len(self._train_lines_np) / 2000
        logger.info("--min_thold: {}".format(min_thold))

        def _train_worker(ctx: dict):
            """单行原始数据的处理执行器"""
            inner_freq_stat_map = collections.defaultdict(int)
            task = ctx.get('task')
            bar: WithMutex = ctx.get('bar')
            worker_id = ctx.get('worker_id')
            line_strs: List[List[str]] = ctx.get('datasets')[worker_id]

            for line in line_strs:
                # 对每个 Byte Pair 进行处理
                for i in range(len(line) - 1):
                    # 如果是 `#`，则跳过
                    if line[i] == '#' or line[i+1] == '#':
                        continue
                    # 获取当前词和下一个词，如 '天安', '门'
                    cur_word: str = line[i]
                    next_word: str = line[i + 1]
                    # 当前词和下一个词拼接，如 '天安门'
                    cur_word_next_word: str = cur_word + next_word
                    # 当前词和下一个词拼接的词频
                    freq = inner_freq_stat_map[cur_word_next_word]
                    # 当前词和下一个词拼接的词频加1
                    inner_freq_stat_map[cur_word_next_word] = freq + 1
                with bar.mutex:
                    bar.obj()

            # 将当前行的词频统计表加入总表
            with shared_freq_stat_map.mutex:
                for key, value in inner_freq_stat_map.items():
                    shared_freq_stat_map.obj[key] += value
        
        def _connect_worker(ctx: dict):
            """分词执行器"""
            bar: WithMutex = ctx.get('bar')
            worker_id = ctx.get('worker_id')
            line_strs: List[List[str]] = ctx.get('datasets')[worker_id]
            thold = ctx.get('thold')

            for line in line_strs:
                # 对每个 Byte Pair 进行处
                i = 0
                while(i < len(line) - 1):
                    # 如果是 `#`，则跳过
                    if line[i] == '#' or line[i+1] == '#':
                        i += 1
                        continue
                    # 获取当前词和下一个词，如 '天安', '门'
                    cur_word: str = line[i]
                    next_word: str = line[i + 1]
                    # 当前词和下一个词拼接，如 '天安门'
                    cur_word_next_word: str = cur_word + next_word
                    # 比较是否大于阈值
                    if shared_freq_stat_map.obj[cur_word_next_word] > thold:
                        if shared_freq_stat_map.obj[cur_word] / shared_freq_stat_map.obj[cur_word_next_word]\
                                > 1.1:
                            i += 1
                            continue
                        # 如果大于阈值，则连接
                        line[i] = cur_word_next_word
                        # 删除下一个词
                        line.pop(i + 1)
                        i += 1
                    i += 1
                with bar.mutex:
                    bar.obj()
        round_num = 1
        max_round_num = 4  # self.max_round_num
        nline = len(self._train_lines_np)
        baseline = int(nline / 5)  # 需调整分母使得ntok_tholds=[4,3,2,1]
        ntok_tholds = [int(baseline*2.4), int(baseline*1.8),
                       int(baseline*1.2), int(baseline*0.8)]
        logger.info("ntok_tholds: {}".format(ntok_tholds))

        while round_num <= max_round_num:
            logger.info("Round {} start...".format(round_num))
            # 初始化
            shared_freq_stat_map = WithMutex(collections.defaultdict(int))

            ParalledTask.create("--freq stat round={}".format(round_num))\
                .set_nworker(self.nworker)\
                .set_worker_args({'datasets': ds})\
                .set_worker_func(_train_worker)\
                .set_progress_goal(len(self._train_lines_np))\
                .execute()

            # 从高频到低频排序
            sorted_freq_stat_ls = sorted(
                shared_freq_stat_map.obj.items(), key=lambda x: x[1], reverse=True)

            # 用 n 个进行分词
            ntok_thold = ntok_tholds[round_num - 1]
            thold = sorted_freq_stat_ls[0:ntok_thold][-1][1]
            thold = max(min_thold, thold)
            logger.info("Thold: {}".format(thold))

            ParalledTask.create("--connect round={}".format(round_num))\
                .set_nworker(self.nworker)\
                .set_worker_args({'datasets': ds, 'thold': thold})\
                .set_worker_func(_connect_worker)\
                .set_progress_goal(len(self._train_lines_np))\
                .execute()

            logger.info("Round {} done.".format(round_num))
            self.dump(self.vocab_file + "_" + str(round_num),
                      sorted_freq_stat_ls, 10000)

            round_num += 1

        # preview 1000 words
        # for i in range(500):
        #     print(sorted_freq_stat_ls[i][0])
    
    def dump(self, filename, freq_map, max_num):
        # 如果文件夹不存在，则创建
        output_dir = os.path.dirname(filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 保存所有词到文件
        with open(filename, 'w', encoding="utf-8") as f:
            i = 0
            for word, freq in freq_map:
                if i >= max_num:
                    break
                f.write(word + '\t' + str(freq) + '\n')
                i += 1
        logger.info("Saved to {}.".format(filename))

def load_word_freq_map(path, thold = 0):
    with open(path, 'r', encoding='utf-8') as f:
        word_freq_map = {}
        for line in f:
            word, freq = line.split('\t')
            freq = int(freq)
            if freq > thold:
                word_freq_map[word] = int(freq)
    return word_freq_map

def tok_worker(ctx: dict):
    bar :WithMutex = ctx.get('bar')
    worker_id = ctx.get('worker_id')
    tokenizer = ctx.get('tokenizer')
    dataset: List[List[str]] = ctx.get('datasets')[worker_id]
    result_lines = []
    for line in dataset:
        result = tokenizer.tokenize(line)
        result_lines.append(result)
        with bar.mutex:
            bar.obj()
    return result_lines

def tok_reducer(results: dict[list]):
    logger = logging.getLogger("default")
    logger.info("Reducing...")
    lines = []
    for k, v in results.items():
        lines.extend(v)
    logger.info("Reduced.")
    return lines

def init_logger(logger):
    """初始化日志记录器，包含时间、日志级别、文件名、行号等详细信息"""
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def split_list(lst, n):
    """将列表 lst 拆分成 n 份"""
    it = iter(lst)
    return [list(islice(it, len(lst) // n + (i < len(lst) % n))) for i in range(n)]

def main():
    args = SimpleNamespace(
        train_file="data/test_sample.txt",
        vocab_file="output/vocab_sample.txt",
        test_file="data/test_sample.txt",
        test_output="output/tokenized.txt",
        nworker=4,
        mode="train_test"
    )
    init_logger(logging.getLogger("default"))
    if args.mode in ['train_test', 'train']:
        bpeCn = BpeCn(args)
        bpeCn.train()
    if args.mode == 'train_test' or args.mode == 'test':
        logger.info("Training done.")
        logger.info("Start testing...")
        test_file = open(args.test_file, 'r', encoding='utf-8').readlines()
        test_lines = [line.strip().replace(' ', '') for line in test_file]
        tokenizer = Tokenizer()\
            .add_dict(load_word_freq_map("output/vocab_sample.txt_1"))\
            .add_dict(load_word_freq_map("output/vocab_sample.txt_2"))\
            .add_dict(load_word_freq_map("output/vocab_sample.txt_3"))\
            .add_dict(load_word_freq_map("output/vocab_sample.txt_4"))
        
        # datasets = np.array_split(test_lines, args.nworker)
        datasets = split_list(test_lines, args.nworker)
        result_lines = ParalledTask.create('-- test')\
            .set_nworker(args.nworker)\
            .set_worker_func(tok_worker)\
            .set_reducer_func(tok_reducer)\
            .set_progress_goal(len(test_lines))\
            .set_worker_args({'datasets': datasets, 'tokenizer': tokenizer})\
            .execute()\
            .get_results()
        
        # print(result_lines)

        with open(args.test_output, 'w', encoding='utf-8') as f:
            for line in result_lines:
                f.write(' '.join(line) + '\n')
        logger.info("Test done. Saved to %s" % args.test_output)

if __name__ == "__main__":
    main()
