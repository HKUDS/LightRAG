"""
Utility function translations
工具函数翻译
"""

UTILITY_TRANSLATIONS = {
    # Environment file checking
    "warning_no_env": [
        "警告: 当前目录中没有找到.env文件，这可能会影响存储配置的加载。",
        "Warning: No .env file found in the current directory, which may affect storage configuration loading.",
    ],
    "continue_execution": [
        "是否继续执行? (yes/no): ",
        "Continue execution? (yes/no): ",
    ],
    "test_cancelled": [
        "测试程序已取消",
        "Test program cancelled",
    ],
    # KuzuDB test environment setup
    "warning_cleanup_temp_dir_failed": [
        "警告: 清理临时目录失败: %s",
        "Warning: Failed to cleanup temporary directory: %s",
    ],
    # Storage initialization errors
    "error_general": [
        "错误: %s",
        "Error: %s",
    ],
    "error_module_path_not_found": [
        "错误: 未找到 %s 的模块路径",
        "Error: Module path not found for %s",
    ],
    "error_import_failed": [
        "错误: 导入 %s 失败: %s",
        "Error: Failed to import %s: %s",
    ],
    "error_initialization_failed": [
        "错误: 初始化 %s 失败: %s",
        "Error: Failed to initialize %s: %s",
    ],
    # New translations for storage setup
    "supported_graph_storage_types": [
        "支持的图存储类型: %s",
        "Supported graph storage types: %s",
    ],
    "error_missing_env_vars": [
        "错误: %s 需要以下环境变量，但未设置: %s",
        "Error: %s requires these environment variables but they are not set: %s",
    ],
    "kuzu_test_environment_setup": [
        "KuzuDB 测试环境已设置 | 测试环境已设置:\n%s",
        "KuzuDB test environment setup | The test environment has been set up:\n%s",
    ],
}
