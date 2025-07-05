# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""简单测试LocalHuggingFaceEmbeddingModel的基本功能."""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.providers.fnllm.models import LocalHuggingFaceEmbeddingModel


def test_model_download_and_embedding():
    """测试模型下载和embedding功能"""
    print("=== 测试LocalHuggingFaceEmbeddingModel ===")

    # 1. 设置模型缓存目录
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HF_DATASETS_CACHE"] = str(models_dir)

    print(f"模型缓存目录: {models_dir}")

    # 2. 创建配置 - 只包含LanguageModelConfig支持的属性
    config = LanguageModelConfig(
        type=ModelType.HuggingFaceLocalEmbedding,
        model="nomic-ai/nomic-embed-text-v1.5",
        api_key="",
        auth_type="api_key",
        encoding_model="cl100k_base"
    )

    print(f"模型名称: {config.model}")
    print(f"模型类型: {config.type}")
    print(f"编码模型: {config.encoding_model}")

    # 3. 创建模型实例（这会触发模型下载）
    print("\n正在创建模型实例...")
    try:
        # 将自定义参数作为kwargs传递
        model = LocalHuggingFaceEmbeddingModel(
            name="test_model",
            config=config,
            # 自定义参数通过kwargs传递
            task_type="search_document",
            matryoshka_dim=768,
            use_sentence_transformers=True
        )
        print("✅ 模型实例创建成功")
    except Exception as e:
        print(f"❌ 模型实例创建失败: {e}")
        return False

    # 4. 测试单个文本embedding
    print("\n正在测试单个文本embedding...")
    test_text = "这是一个测试文本，用于验证embedding功能。"
    print(f"测试文本: {test_text}")

    try:
        embedding = model.embed(test_text)
        print(f"✅ 单个文本embedding成功")
        print(f"   Embedding维度: {len(embedding)}")
        print(f"   前5个值: {embedding[:5]}")
        print(f"   后5个值: {embedding[-5:]}")
    except Exception as e:
        print(f"❌ 单个文本embedding失败: {e}")
        return False

    # 5. 测试批量文本embedding
    print("\n正在测试批量文本embedding...")
    test_texts = [
        "第一个测试文本",
        "第二个测试文本，内容更长一些",
        "第三个测试文本，包含特殊字符：@#$%^&*()"
    ]
    print(f"测试文本列表: {test_texts}")

    try:
        embeddings = model.embed_batch(test_texts)
        print(f"✅ 批量文本embedding成功")
        print(f"   文本数量: {len(embeddings)}")
        print(f"   每个embedding维度: {len(embeddings[0])}")
        print(f"   第一个embedding前5个值: {embeddings[0][:5]}")
    except Exception as e:
        print(f"❌ 批量文本embedding失败: {e}")
        return False

    # 6. 检查模型文件是否下载
    print(f"\n检查模型文件...")
    if models_dir.exists():
        model_files = list(models_dir.rglob("*"))
        print(f"   模型目录中的文件数量: {len(model_files)}")
        if model_files:
            print(f"   前5个文件: {[f.name for f in model_files[:5]]}")
    else:
        print("   模型目录不存在")

    print("\n🎉 所有测试都通过了！")
    print("✅ 模型下载成功")
    print("✅ 单个文本embedding成功")
    print("✅ 批量文本embedding成功")

    return True


if __name__ == "__main__":
    print("开始测试LocalHuggingFaceEmbeddingModel...")
    print("=" * 60)

    success = test_model_download_and_embedding()

    if success:
        print("\n" + "=" * 60)
        print("🎉 测试完成！LocalHuggingFaceEmbeddingModel工作正常。")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败！请检查错误信息。")
        print("=" * 60)