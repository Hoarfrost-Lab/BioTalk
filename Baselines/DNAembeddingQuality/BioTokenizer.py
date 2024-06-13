from transformers import RobertaTokenizer
class BioTokenizer(RobertaTokenizer):
  def __init__(self, ksize=1, stride=1, include_bos=False, include_eos=False, **kwargs):
    super().__init__(**kwargs)
    self.ksize = ksize
    self.stride = stride
    self.include_bos = include_bos
    self.include_eos = include_eos
  def tokenize(self, t, **kwargs):
    include_bos = self.include_bos if self.include_bos is not None else include_bos
    include_eos = self.include_eos if self.include_eos is not None else include_eos
    t = t.upper()
    if self.ksize == 1:
        toks = list(t)
#    else:
#        toks = [t[i:i + self.ksize] for i in range(0, len(t), self.stride) if len(t[i:i + self.ksize]) == self.ksize]
    if len(toks[-1]) < self.ksize:
        toks = toks[:-1]
    if include_bos:
        toks = ['S'] + toks
    if include_eos:
        toks = toks + ['/S']
    return toks
