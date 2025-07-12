"""
Common translations used across multiple tests
"""

COMMON_TRANSLATIONS = {
    # Program header
    "program_title": [
        """
    ╔══════════════════════════════════════════════════════════════╗
    ║                     通用图存储测试程序                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """,
        """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 Universal Graph Storage Test                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """,
    ],
    # Basic elements
    "node": ["节点", "Node"],
    "edge": ["边", "Edge"],
    "degree": ["度数", "Degree"],
    "delete_edge": ["删除边", "Delete Edge"],
    # Basic operations
    "insert_node": ["插入节点", "Insert node"],
    "insert_edge": ["插入边", "Insert edge"],
    "read_node_props": ["读取节点属性", "Read node properties"],
    "read_edge_props": ["读取边属性", "Read edge properties"],
    "read_reverse_edge": ["读取反向边属性", "Read reverse edge properties"],
    # Entity data
    "artificial_intelligence": ["人工智能", "Artificial Intelligence"],
    "machine_learning": ["机器学习", "Machine Learning"],
    "deep_learning": ["深度学习", "Deep Learning"],
    "natural_language_processing": ["自然语言处理", "Natural Language Processing"],
    "computer_vision": ["计算机视觉", "Computer Vision"],
    "computer_science": ["计算机科学", "Computer Science"],
    "data_structure": ["数据结构", "Data Structure"],
    "algorithm": ["算法", "Algorithm"],
    # Descriptions
    "ai_desc": [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "Artificial Intelligence is a branch of computer science that attempts to understand the essence of intelligence and produce a new kind of intelligent machine that can respond in a way similar to human intelligence.",
    ],
    "ml_desc": [
        "机器学习是人工智能的一个分支，它使用统计学方法让计算机系统在不被明确编程的情况下也能够学习。",
        "Machine Learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
    ],
    "dl_desc": [
        "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。",
        "Deep Learning is a branch of machine learning that uses multi-layer neural networks to simulate the human brain's learning process.",
    ],
    "nlp_desc": [
        "自然语言处理是人工智能的一个分支，专注于使计算机理解和处理人类语言。",
        "Natural Language Processing is a branch of artificial intelligence that focuses on enabling computers to understand and process human language.",
    ],
    "cv_desc": [
        "计算机视觉是人工智能的一个分支，专注于使计算机能够从图像或视频中获取信息。",
        "Computer Vision is a branch of artificial intelligence that focuses on enabling computers to obtain information from images or videos.",
    ],
    "cs_desc": [
        "计算机科学是研究计算机及其应用的科学。",
        "Computer Science is the science that studies computers and their applications.",
    ],
    "ds_desc": [
        "数据结构是计算机科学中的一个基础概念，用于组织和存储数据。",
        "Data Structure is a fundamental concept in computer science, used to organize and store data.",
    ],
    "algo_desc": [
        "算法是解决问题的步骤和方法。",
        "Algorithm is the steps and methods for solving problems.",
    ],
    # Keywords
    "ai_keywords": ["AI,机器学习,深度学习", "AI,machine learning,deep learning"],
    "ml_keywords": [
        "监督学习,无监督学习,强化学习",
        "supervised learning,unsupervised learning,reinforcement learning",
    ],
    "dl_keywords": ["神经网络,CNN,RNN", "neural networks,CNN,RNN"],
    "node_degree_should_be": [
        "节点 %s 的度数应为%d，实际为 %d",
        "Node %s degree should be %d, actual %d",
    ],
    "node_should_have_edges": [
        "节点 %s 应有%d条边，实际有 %d",
        "Node %s should have %d edges, actual %d",
    ],
    "undirected_verification_success": [
        "无向图特性验证成功：正向和反向边属性一致",
        "Undirected graph property verification successful: forward and reverse edge properties are consistent",
    ],
    "nlp_keywords": ["NLP,文本分析,语言模型", "NLP,text analysis,language models"],
    "cv_keywords": ["CV,图像识别,目标检测", "CV,image recognition,object detection"],
    "cs_keywords": ["计算机,科学,技术", "computer,science,technology"],
    "ds_keywords": ["数据,结构,组织", "data,structure,organization"],
    "algo_keywords": ["算法,步骤,方法", "algorithm,steps,methods"],
    # Entity types
    "tech_field": ["技术领域", "Technology Field"],
    "concept": ["概念", "Concept"],
    "subject": ["学科", "Subject"],
    "test_node": ["测试节点", "Test Node"],
    # Relationships
    "contains": ["包含", "contains"],
    "applied_to": ["应用于", "applied to"],
    "special_relation": ["特殊'关系'", "special 'relation'"],
    # Relationship descriptions
    "ai_contains_ml": [
        "人工智能领域包含机器学习这个子领域",
        "The field of artificial intelligence contains machine learning as a subfield",
    ],
    "ml_contains_dl": [
        "机器学习领域包含深度学习这个子领域",
        "The field of machine learning contains deep learning as a subfield",
    ],
    "ai_contains_nlp": [
        "人工智能领域包含自然语言处理这个子领域",
        "The field of artificial intelligence contains natural language processing as a subfield",
    ],
    "ai_contains_cv": [
        "人工智能领域包含计算机视觉这个子领域",
        "The field of artificial intelligence contains computer vision as a subfield",
    ],
    "ai_contains_cv_desc": [
        "人工智能领域包含计算机视觉这个子领域",
        "The field of artificial intelligence contains computer vision as a subfield",
    ],
    "dl_applied_nlp": [
        "深度学习技术应用于自然语言处理领域",
        "Deep learning technology is applied to the field of natural language processing",
    ],
    "dl_applied_nlp_relationship": ["应用于", "applied to"],
    "dl_applied_nlp_desc": [
        "深度学习技术应用于自然语言处理领域",
        "Deep learning technology is applied to the field of natural language processing",
    ],
    "dl_applied_cv": [
        "深度学习技术应用于计算机视觉领域",
        "Deep learning technology is applied to the field of computer vision",
    ],
    "dl_applied_cv_relationship": ["应用于", "applied to"],
    "dl_applied_cv_desc": [
        "深度学习技术应用于计算机视觉领域",
        "Deep learning technology is applied to the field of computer vision",
    ],
    "cs_contains_ds": [
        "计算机科学包含数据结构这个概念",
        "Computer science contains data structures as a concept",
    ],
    "cs_contains_algo": [
        "计算机科学包含算法这个概念",
        "Computer science contains algorithms as a concept",
    ],
    # Common status messages
    "no_description": ["无描述", "No description"],
    "no_type": ["无类型", "No type"],
    "no_keywords": ["无关键词", "No keywords"],
    "no_relationship": ["无关系", "No relationship"],
    "no_weight": ["无权重", "No weight"],
    "actual": ["实际", "actual"],
    "test_error": ["测试过程中发生错误", "Error occurred during testing"],
    # System messages
    "error": ["错误", "Error"],
    "warning": ["警告", "Warning"],
    "current_graph_storage": [
        "当前配置的图存储类型",
        "Current configured graph storage type",
    ],
    "supported_graph_storage": [
        "支持的图存储类型",
        "Supported graph storage types",
    ],
    "init_storage_failed": [
        "初始化存储实例失败，测试程序退出",
        "Failed to initialize storage instance, test program exiting",
    ],
    # Additional node operations
    "insert_node_1": ["插入节点1", "Insert node 1"],
    "insert_node_2": ["插入节点2", "Insert node 2"],
    "insert_node_3": ["插入节点3", "Insert node 3"],
    "insert_edge_1": ["插入边1", "Insert edge 1"],
    "insert_edge_2": ["插入边2", "Insert edge 2"],
    # Edge properties
    "edge_desc": ["边描述", "Edge description"],
    "edge_relationship": ["边关系", "Edge relationship"],
    "edge_weight": ["边权重", "Edge weight"],
    # Common descriptions
    "no_description": ["无描述", "No description"],
    "no_relationship": ["无关系", "No relationship"],
    "batch_get_edges_result": [
        "批量获取边属性结果",
        "Batch get edge properties result",
    ],
    "batch_get_reverse_edges_result": [
        "批量获取反向边属性结果",
        "Batch get reverse edge properties result",
    ],
    "batch_get_nodes_edges_result": [
        "批量获取节点边结果",
        "Batch get node edges result",
    ],
    "undirected_nodes_edges_verification_success": [
        "无向图特性验证成功：批量获取的节点边包含所有相关的边（无论方向）",
        "Undirected graph property verification successful: batch obtained node edges contain all related edges (regardless of direction)",
    ],
}
