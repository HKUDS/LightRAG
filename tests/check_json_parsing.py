import sys
import json
import re

def parse_llm_response(response_text):
    """安全地解析LLM的JSON响应，处理各种格式和边缘情况。"""
    if not response_text: 
        return None
    
    print(f"原始响应: {response_text}")
    
    try:
        # 1. 清理Markdown代码块标记
        cleaned_text = response_text.strip()
        
        # 特别处理包含```json的代码块 (常见于大部分LLM响应)
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        json_blocks = re.findall(json_block_pattern, cleaned_text, re.DOTALL)
        
        if json_blocks:
            # 如果有多个代码块，选择最长的一个
            longest_block = max(json_blocks, key=len).strip()
            print(f"从Markdown代码块提取JSON: {longest_block}")
            cleaned_text = longest_block
        
        # 2. 尝试直接解析清理后的文本
        try:
            parsed_data = json.loads(cleaned_text)
            if isinstance(parsed_data, dict) and \
               (('entities' in parsed_data and isinstance(parsed_data['entities'], list)) or \
                ('relations' in parsed_data and isinstance(parsed_data['relations'], list))):
                print("成功直接解析JSON响应")
                return parsed_data
        except json.JSONDecodeError as e:
            # 如果直接解析失败，继续使用更复杂的方法
            print(f"直接解析JSON失败: {e}，尝试修复...")
        
        # 3. 定位JSON对象的边界
        # 查找第一个左大括号和最后一个右大括号
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            # 提取潜在的JSON文本
            json_text = cleaned_text[json_start:json_end + 1]
            print(f"提取潜在JSON文本: {json_text}")
            
            # 4. 尝试解析提取的JSON
            try:
                parsed_data = json.loads(json_text)
                if isinstance(parsed_data, dict) and \
                   (('entities' in parsed_data and isinstance(parsed_data['entities'], list)) or \
                    ('relations' in parsed_data and isinstance(parsed_data['relations'], list))):
                    print("成功解析提取的JSON文本")
                    return parsed_data
            except json.JSONDecodeError as e:
                # 如果仍然失败，尝试更高级的修复
                print(f"解析提取的JSON文本失败: {e}")
            
            # 5. 尝试修复可能不完整的JSON
            # 查找实体或关系数组
            try:
                if '"entities"' in json_text:
                    entities_match = re.search(r'"entities"\s*:\s*\[(.*?)(?:\]|$)', json_text, re.DOTALL)
                    if entities_match:
                        entities_content = entities_match.group(1).strip()
                        print(f"提取entities内容: {entities_content}")
                        # 检查最后一个对象是否完整
                        if entities_content.endswith(','):
                            entities_content = entities_content[:-1]  # 移除尾部逗号
                        
                        # 确保JSON数组内容有效
                        if not entities_content.endswith('}'):
                            # 查找最后一个完整的对象
                            last_complete_obj = entities_content.rfind('}')
                            if last_complete_obj != -1:
                                entities_content = entities_content[:last_complete_obj+1]
                                print(f"截取到最后一个完整对象: {entities_content}")
                        
                        # 构建完整的JSON字符串
                        fixed_json = f'{{"entities": [{entities_content}]}}'
                        try:
                            parsed_data = json.loads(fixed_json)
                            print(f"成功修复并解析entities JSON: {fixed_json}")
                            return parsed_data
                        except json.JSONDecodeError as e:
                            print(f"修复entities JSON失败: {e}")
                
            except Exception as repair_error:
                print(f"尝试修复JSON时出错: {repair_error}")
        
        # 6. 使用正则表达式直接提取实体或关系对象
        try:
            # 匹配实体对象
            entity_pattern = r'\{\s*"name"\s*:\s*"((?:\\.|[^"\\])*?)"\s*,\s*"type"\s*:\s*"((?:\\.|[^"\\])*?)"\s*\}'
            entities = re.findall(entity_pattern, cleaned_text)
            if entities:
                entity_objects = [{"name": name, "type": type_} for name, type_ in entities]
                print(f"通过正则表达式提取了 {len(entity_objects)} 个实体: {entity_objects}")
                return {"entities": entity_objects}
        except Exception as regex_error:
            print(f"使用正则表达式提取JSON时出错: {regex_error}")
        
        # 所有方法都失败，记录警告并输出完整响应
        print(f"无法从LLM响应中提取有效的JSON结构:\n{response_text}")
        return None
    except Exception as e:
        print(f"解析LLM响应时发生未预期的错误: {e}")
        print(f"问题响应完整内容:\n{response_text}")
        return None

def main():
    # 测试不完整的JSON响应
    test_response = '''```json
{
  "entities": [
    {"name": "广州局集团公司", "type": "Organization"},
    {"name": "集团公司", "type": "Organization"},
    {"name": "国铁集团", "type": "Organization"},
    {"name": "信息技术所", "type": "Organization"}
  ]
}```'''

    result = parse_llm_response(test_response)
    print("\n解析结果:", result)
    
    # 测试截断的JSON响应1：末尾实体被截断
    truncated_response1 = '''```json
{
  "entities": [
    {"name": "广州局集团公司", "type": "Organization"},
    {"name": "集团公司", "type": "Organization"},
    {"name": "国铁集团", "type": "Organization"},
    {"name": "信息技术所", "type": "Or...'''
    
    print("\n\n测试截断的JSON响应1（末尾实体被截断）:")
    result = parse_llm_response(truncated_response1)
    print("\n解析结果:", result)
    
    # 测试截断的JSON响应2：模拟日志中看到的截断
    truncated_response2 = '''```json
{
  "entities": [
    {"name": "广州局集团公司", "type": "Organization"},
    {"name": "集团公司", "type": "Organization"},
    {"name": "国铁集团", "type": "Organization"},
    {"name": "信息技术所", "type": "Organization"},
    {"name": "系统维护、管理", "type": "Statement"},
    {"name": "网络管理岗", "type": "Role"},
    {"name": "安全管理岗", "type": "Role"},
    {"name": "信息安全管理制度", "type": "Statement"},
    {"name": "广州铁路", "type": "Organization"},
    {"name": "行业网维护管理部门", "type": "Organization"},
    {"name": "防灾安全监控系统机房", "type": "Topic"},
    {"name": "系统集成商", "type": "Organization"},
    {"name": "网络安全管理规范", "type": "Statement"},
    {"name": "客服客票系统", "type": "Topic"},
    {"name": "铁路货运系统", "type": "Topic"},
    {"name": "信息系统安全", "type": "Topic"'''
    
    print("\n\n测试截断的JSON响应2（模拟日志中看到的截断）:")
    result = parse_llm_response(truncated_response2)
    print("\n解析结果:", result)
    
    # 测试截断的JSON响应3：完全符合错误日志中的JSON
    truncated_response3 = '''```json
{
  "entities": [
    {"name": "广州局集团公司", "type": "Organization"},
    {"name": "集团公司", "type": "Organization"},
    {"name": "国铁集团", "type": "Organization"},
    {"name": "信息技术所", "type": "Or...'''
    
    print("\n\n测试截断的JSON响应3（完全符合错误日志中的JSON）:")
    result = parse_llm_response(truncated_response3)
    print("\n解析结果:", result)
    
    # 测试正则表达式方法
    truncated_response4 = '''```json
{
  "entities": [
    {"name": "广州局集团公司", "type": "Organization"},
    {"name": "集团公司", "type": "Organization"},
    {"name": "国铁集团", "type": "Organization"},
    {"name": "信息技术所", "type": "Or'''
    
    print("\n\n测试截断的JSON响应4（正则表达式提取）:")
    
    # 临时修改parse_llm_response函数来强制使用正则表达式方法
    def test_regex_only(response_text):
        if not response_text: 
            return None
        
        print(f"原始响应: {response_text}")
        
        # 直接尝试正则表达式提取
        entity_pattern = r'\{\s*"name"\s*:\s*"((?:\\.|[^"\\])*?)"\s*,\s*"type"\s*:\s*"((?:\\.|[^"\\])*?)"\s*\}'
        entities = re.findall(entity_pattern, response_text)
        
        if entities:
            entity_objects = [{"name": name, "type": type_} for name, type_ in entities]
            print(f"通过正则表达式提取了 {len(entity_objects)} 个实体: {entity_objects}")
            return {"entities": entity_objects}
        
        return None
    
    regex_result = test_regex_only(truncated_response4)
    print("\n正则表达式提取结果:", regex_result)

if __name__ == "__main__":
    main() 