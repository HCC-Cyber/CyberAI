#!/usr/bin/env python3
"""
SenseVoiceSmall 模型下载器
功能：独立下载和管理语音识别模型
"""


import os
from modelscope import snapshot_download
import warnings

from utils.util import Util
warnings.filterwarnings('ignore')

class ModelDownloader:
    """模型下载管理器"""
    
    def __init__(self, model_id='iic/SenseVoiceSmall', local_dir=None):
        """
        初始化下载器
        
        参数:
            model_id: ModelScope模型ID
            local_dir: 本地保存目录（默认为 Util.get_process_dir() + "ai_core/asr/funasr/model"）
        """
        self.model_id = model_id
        
        # 设置默认本地目录为 Util.get_process_dir() + "ai_core" + os.sep + "asr" + os.sep + "funasr" + os.sep + "model"
        if local_dir is None:
            process_dir = Util.get_process_dir()
            self.local_dir = os.path.join(process_dir, "ai_core", "asr", "funasr", "model")
        else:
            self.local_dir = local_dir
        
        # 从model_id提取模型名称
        self.model_name = model_id.split('/')[-1]
    
    def check_model_exists(self):
        """
        检测模型是否已存在
        
        返回值:
            bool: 模型存在返回True，否则返回False
        """
        # 检查模型目录是否存在
        if not os.path.exists(self.local_dir):
            print(f"⚠️  模型目录不存在: {self.local_dir}")
            return False
        
        # 检查模型目录是否为空
        if not os.listdir(self.local_dir):
            print(f"⚠️  模型目录为空: {self.local_dir}")
            return False
        
        # 检查关键模型文件是否存在
        required_files = [
            'configuration.json',
            'model.safetensors',  # 优先检查safetensors格式
            'pytorch_model.bin',  # 备选模型文件
            'model.pt'  # 备选模型文件
        ]
        
        # 检查是否存在至少一个模型文件
        model_files_found = []
        for file in os.listdir(self.local_dir):
            if any(req_file in file for req_file in ['model.safetensors', 'pytorch_model.bin', 'model.pt']):
                model_files_found.append(file)
        
        if not model_files_found:
            print(f"⚠️  未找到模型文件 (model.safetensors, pytorch_model.bin, model.pt) 在: {self.local_dir}")
            return False
        
        print(f"✓ 模型已存在: {self.local_dir}")
        print(f"  找到模型文件: {model_files_found}")
        return True
    
    def download(self, force_download=False):
        """
        下载模型到本地目录
        
        参数:
            force_download: 是否强制重新下载
            
        返回值:
            模型本地路径
        """
        print(f"正在下载模型: {self.model_id}")
        print(f"模型名称: {self.model_name}")
        print(f"保存目录: {self.local_dir}")
        
        # 如果模型已存在且不强制重新下载，直接返回
        if not force_download and self.check_model_exists():
            print("✓ 模型已存在，跳过下载")
            self.model_dir = self.local_dir
            return self.local_dir
        
        try:
            # 确保本地目录存在
            os.makedirs(self.local_dir, exist_ok=True)
            
            # 下载模型到指定本地目录
            self.model_dir = snapshot_download(
                model_id=self.model_id,
                revision='master',
                local_dir=self.local_dir,
                cache_dir=None  # 不使用缓存，直接下载到指定目录
            )
            
            print(f"✓ 模型下载成功！")
            print(f"本地路径: {self.model_dir}")
            
            # 检查模型文件
            self._check_model_files()
            
            return self.model_dir
            
        except Exception as e:
            print(f"✗ 模型下载失败: {e}")
            return None
    
    def _check_model_files(self):
        """检查模型文件完整性"""
        if not self.model_dir or not os.path.exists(self.model_dir):
            print("⚠️  模型目录不存在")
            return False
        
        print("\n📁 模型文件结构:")
        files_found = []
        
        # 检查常见模型文件
        expected_files = [
            'configuration.json',
            'model.safetensors',
            'model.pt',
            'pytorch_model.bin',
            'preprocessor_config.json',
            'config.yaml',
            'tokenizer.json',
            'vocab.txt'
        ]
        
        for file in os.listdir(self.model_dir):
            files_found.append(file)
            if any(expected in file for expected in expected_files):
                print(f"  ✓ {file}")
            else:
                print(f"    {file}")
        
        # 统计文件数量
        print(f"\n📊 文件统计: {len(files_found)} 个文件/目录")
        
        return True
    
    def get_model_info(self):
        """获取模型信息"""
        if not self.model_dir:
            return None
        
        info = {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'local_path': self.model_dir,
            'file_count': len(os.listdir(self.model_dir)) if os.path.exists(self.model_dir) else 0,
            'total_size': self._get_folder_size(self.model_dir) if os.path.exists(self.model_dir) else 0,
            'exists': self.check_model_exists()
        }
        
        return info
    
    def _get_folder_size(self, folder_path):
        """计算文件夹大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        
        # 转换为MB
        return total_size / (1024 * 1024)
    
    def set_local_dir(self, local_dir):
        """设置本地保存目录"""
        self.local_dir = local_dir
        print(f"本地保存目录已设置为: {local_dir}")

# 独立下载函数（兼容旧版本）
def download_sensevoice_model(model_id='iic/SenseVoiceSmall', local_dir=None, force=False):
    """
    下载SenseVoice模型的简化函数
    
    参数:
        model_id: 模型ID
        local_dir: 本地保存目录
        force: 是否强制重新下载
        
    返回值:
        模型本地路径
    """
    downloader = ModelDownloader(model_id, local_dir)
    return downloader.download(force_download=force)

# 检测模型是否存在
def check_model_exists(model_id='iic/SenseVoiceSmall', local_dir=None):
    """
    检测模型是否存在
    
    参数:
        model_id: 模型ID
        local_dir: 本地保存目录
        
    返回值:
        bool: 模型存在返回True，否则返回False
    """
    downloader = ModelDownloader(model_id, local_dir)
    return downloader.check_model_exists()

# 命令行支持
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载SenseVoiceSmall语音识别模型')
    parser.add_argument('--model-id', type=str, default='iic/SenseVoiceSmall',
                       help='ModelScope模型ID (默认: iic/SenseVoiceSmall)')
    parser.add_argument('--local-dir', type=str, default=None,
                       help='本地保存目录 (默认: Util.get_process_dir()/ai_core/asr/funasr/model)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新下载')
    parser.add_argument('--info', action='store_true',
                       help='显示模型信息')
    parser.add_argument('--check', action='store_true',
                       help='检测模型是否存在')
    
    args = parser.parse_args()
    
    # 如果没有指定本地目录，使用 Util.get_process_dir() + "ai_core/asr/funasr/model"
    if args.local_dir is None:
        process_dir = Util.get_process_dir()
        args.local_dir = os.path.join(process_dir, "ai_core", "asr", "funasr", "model")
    
    # 创建下载器实例
    downloader = ModelDownloader(args.model_id, args.local_dir)
    
    if args.check:
        # 检测模型是否存在
        exists = downloader.check_model_exists()
        print(f"\n📋 模型存在状态: {'✓ 存在' if exists else '✗ 不存在'}")
        print(f"模型路径: {downloader.local_dir}")
    elif args.info and os.path.exists(downloader.local_dir) and downloader.check_model_exists():
        # 显示现有模型信息
        info = downloader.get_model_info()
        if info:
            print("\n📋 模型信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    else:
        # 下载模型
        model_path = downloader.download(force_download=args.force)
        
        if model_path:
            # 显示下载完成信息
            print("\n🎉 下载完成！")
            print(f"模型已保存到: {model_path}")
            print("\n使用方法:")
            print(f"from download import ModelDownloader")
            print(f"downloader = ModelDownloader('{args.model_id}')")
            print(f"model_path = downloader.download()")