
# Cypher查询语句示例

## 基本查询
```cypher
// 查询所有Action节点
MATCH (n:Action) RETURN n LIMIT 10;

// 查询所有Resource节点
MATCH (n:Resource) RETURN n.name, n.uuid LIMIT 20;

// 统计各类型节点数量
MATCH (n) RETURN DISTINCT labels(n) AS 节点类型, COUNT(*) AS 数量;
```

## 属性过滤查询
```cypher
// 查找名称包含"应急"的所有节点
MATCH (n) WHERE n.name CONTAINS '应急' RETURN n.name, labels(n) AS 类型;

// 查找特定chunk_id的节点
MATCH (n) WHERE n.chunk_id = '76848769_chunk_3_9' RETURN n;

// 使用正则表达式查询
MATCH (n:Action) WHERE n.name =~ '.*12306.*' RETURN n.name;
```

## 关系查询和路径查询
```cypher
// 创建并查询两个节点间的关系
MATCH (a:Action), (r:Resource) 
WHERE a.name CONTAINS '客票' AND r.name CONTAINS '应急预案'
CREATE (a)-[rel:USES]->(r)
RETURN a.name, type(rel), r.name;

// 查询两跳以内的所有路径
MATCH path = (a:Action)-[*1..2]-(b)
WHERE a.name = '开展应急处置工作'
RETURN path LIMIT 10;
```

## 聚合查询
```cypher
// 按chunk_id分组统计节点数量
MATCH (n) 
RETURN n.chunk_id AS 文档ID, COUNT(*) AS 节点数量
ORDER BY 节点数量 DESC LIMIT 10;

// 查找出现最多的Action类型
MATCH (n:Action)
RETURN n.name AS 行动, COUNT(*) AS 出现次数
ORDER BY 出现次数 DESC LIMIT 5;
```

## 高级查询和数据操作
```cypher
// 使用CALL子句执行存储过程
CALL db.labels() YIELD label
RETURN label ORDER BY label;

// WITH子句进行查询链接
MATCH (n:Action)
WITH COLLECT(n) AS actions
MATCH (r:Resource)
RETURN SIZE(actions) AS action数量, COUNT(r) AS resource数量;

// 条件更新节点属性
MATCH (n:Resource)
WHERE n.name = '应急预案'
SET n.importance = 'high'
RETURN n;

// 删除特定条件的重复节点(保留一个)
MATCH (n:Action)
WITH n.name AS name, COLLECT(n) AS nodes
WHERE SIZE(nodes) > 1
WITH nodes[0] AS keepNode, nodes[1..] AS removeNodes
FOREACH (node IN removeNodes | DETACH DELETE node)
RETURN count(removeNodes) AS 已删除节点数;
```

## 复杂组合查询
```cypher
// 查找与特定Action相关的所有资源和组织
MATCH (a:Action)-[r1]-(x)-[r2]-(o:Organization)
WHERE a.name CONTAINS '应急处置'
RETURN a.name AS 行动, type(r1) AS 关系1, x.name AS 中间节点, 
       type(r2) AS 关系2, o.name AS 组织机构;

// 使用UNION组合多个查询结果
MATCH (a:Action) WHERE a.name CONTAINS '旅客'
RETURN a.name AS 名称, '行动' AS 类型
UNION
MATCH (r:Resource) WHERE r.name CONTAINS '系统'
RETURN r.name AS 名称, '资源' AS 类型;

// 使用CASE进行条件处理
MATCH (n) 
RETURN n.name, 
       CASE 
         WHEN n:Action THEN '行动节点'
         WHEN n:Resource THEN '资源节点'
         WHEN n:Organization THEN '组织节点'
         ELSE '其他节点'
       END AS 节点类型
LIMIT 20;
```

## 全文搜索和模糊匹配
```cypher
// 创建全文索引(如果支持)
CREATE FULLTEXT INDEX node_name_index FOR (n) ON EACH [n.name];

// 使用全文搜索
CALL db.index.fulltext.queryNodes("node_name_index", "应急 客票") 
YIELD node, score
RETURN labels(node)[0] AS 类型, node.name AS 名称, score AS 相关度
ORDER BY 相关度 DESC
LIMIT 10;
```

以上Cypher查询语句展示了多种Neo4j图数据库的查询和操作功能，可根据实际需求进行调整。
