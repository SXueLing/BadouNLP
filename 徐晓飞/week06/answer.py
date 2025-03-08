# embding
sentence_length = 9
vocabulary_him = 768

word_him = 256

embding_parameters = vocabulary_him * word_him + (sentence_length * word_him * 2 + 2 * word_him )
self_attention = sentence_length * word_him * 768 * 3
feed = sentence_length * 768 *2

parameter = embding_parameters + self_attention + feed
