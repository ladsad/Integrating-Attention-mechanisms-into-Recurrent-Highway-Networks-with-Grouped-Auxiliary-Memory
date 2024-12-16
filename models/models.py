from models.gam_rhn import GAM_RHN
from models.gam_rhn_attention import GAM_RHN_Attention
import config

def get_baseline_model():
    return GAM_RHN(config.VOCAB_SIZE, config.EMBEDDING_DIM, config.HIDDEN_SIZE, config.NUM_CLASSES).to(config.DEVICE)

def get_attention_model(attention_type='dot_product'):
    return GAM_RHN_Attention(config.VOCAB_SIZE, config.EMBEDDING_DIM, config.HIDDEN_SIZE, config.NUM_CLASSES, attention_type, config.NUM_HEADS).to(config.DEVICE)