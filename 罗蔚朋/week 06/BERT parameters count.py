class Embedding:
    vocab_size = 30522
    hidden_size = 768
    position_embedding = 512
    segment_embedding = 2

class Multi_Heads:
    hidden_size = 768
    head_count = hidden_size / 12
    inner_parameter = 3

class Add_Norm:
    hidden_size = 768

class Feed_Forward:
    hidden_size = 768
    w2 = hidden_size * 4

class Layer_Norm:
    hidden_size = 768

class calculation (Embedding, Multi_Heads, Add_Norm, Feed_Forward, Layer_Norm):
    def Embedding_count (self):
        self.embedding_parameters = (self.vocab_size + self.position_embedding + self.segment_embedding) * self.hidden_size
        return self.embedding_parameters

    def Multi_Heads_count (self):
        self.single_head_parameters = self.hidden_size * (self.hidden_size/12) *self.inner_parameter
        self.link_heads_parameters = 12 * (self.hidden_size/12) *self.hidden_size
        self.multi_heads_parameters = ((self.single_head_parameters *12) + self.link_heads_parameters)*12
        return self.multi_heads_parameters

    def Add_Norm_count (self):
        self.add_norm_parameters = self.hidden_size * 2 * 12
        return self.add_norm_parameters

    def Feed_Forward_count (self):
        self.feedforward_parameters = self.hidden_size * self.w2 * 2 * 12
        return self.feedforward_parameters

    def Layer_Norm_count (self):
        self.layer_norm_parameters = self.hidden_size * 2 * 12
        return self.layer_norm_parameters
    
c=calculation()
embedding_parameters = c.Embedding_count()
print('embedding_parameters:' , embedding_parameters)
multiheads_parameters = c.Multi_Heads_count()
print('multiheads_parameters:' , multiheads_parameters)
addnorm_parameters = c.Add_Norm_count()
print('add_norm_parameters:' , addnorm_parameters)
feedforward_parameters = c.Feed_Forward_count()
print('feed_forward_parameters:' , feedforward_parameters)
layernorm_parameters = c.Layer_Norm_count()
print('layernorm_parameters:' , layernorm_parameters)
total_parameters = embedding_parameters + multiheads_parameters + addnorm_parameters + feedforward_parameters + layernorm_parameters
print('total parameters number:', total_parameters)


#embedding_parameters: 23835648
#multiheads_parameters: 28311552.0
#add_norm_parameters: 18432
#feed_forward_parameters: 56623104
#layernorm_parameters: 18432
#total parameters number: 108807168.0
