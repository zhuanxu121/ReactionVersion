from chem_mech.core.engine import MECH_REGISTRY
from .roles import SN2Plugin

# 注册 SN2 插件
MECH_REGISTRY['sn2'] = SN2Plugin
