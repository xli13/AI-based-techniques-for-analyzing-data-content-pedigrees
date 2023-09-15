import pkuseg
import sys
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
from Crypto import Random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PrpCrypt(object):

  def __init__(self, key):
    self.key = key.encode('utf-8')
    self.mode = AES.MODE_CBC
    self.iv = Random.new().read(AES.block_size)

  def encrypt(self, text):
    text = text.encode('utf-8')
    cryptor = AES.new(self.key, self.mode,self.iv)
    length = 16
    count = len(text)
    if count < length:
      add = (length - count)
      text = text + ('\0' * add).encode('utf-8')
    elif count > length:
      add = (length - (count % length))
      text = text + ('\0' * add).encode('utf-8')
    self.ciphertext = cryptor.encrypt(text)
    return b2a_hex(self.ciphertext)

  def decrypt(self, text):
    cryptor = AES.new(self.key, self.mode, self.iv)
    plain_text = cryptor.decrypt(a2b_hex(text))
    return bytes.decode(plain_text).rstrip('\0')



def encrypt_encrypt(corpus_vec):
    pc = PrpCrypt('0CoJUm6Qyw8W8jud')

    # 加密过程
    encrypt_corpus_vec=[]
    for ori_vec in corpus_vec:
        encrypt_ori_vec=[]
        for v in ori_vec:
            e = pc.encrypt(str(v))
            encrypt_ori_vec.append(e)
        encrypt_corpus_vec.append(encrypt_ori_vec)

    # 解密过程
    decrypt_corpus_vec=[]
    for encrypt_ori_vec in encrypt_corpus_vec:
        decrypt_ori_vec=[]
        for ev in encrypt_ori_vec:
            d = pc.decrypt(ev).encode()
            decrypt_ori_vec.append(float(d))
        decrypt_corpus_vec.append(decrypt_ori_vec)

    return decrypt_corpus_vec
