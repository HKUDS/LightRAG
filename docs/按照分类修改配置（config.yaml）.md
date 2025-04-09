# 修改配置

## 1. 修改配置

**system**: 

```
You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.
```

**user**: 

```
参考我给你的指南，修改congih_simple.yaml，按照不同分类增加下面的schema 和提示词模板
======================
General_Principles_Definitions (总则与定义)
内容: 规章的制定目的、依据、适用范围、基本原则、术语定义、整体目标等。
抽取侧重: Purpose, Scope (适用对象: Organization, Location, System), GuidingPrinciple, Definition。
Organizational_Responsibilities (组织与职责)
内容: 明确各部门、岗位、单位或相关方的管理职责、权限、协作关系、领导关系等。
抽取侧重: Organization, Role, Responsibility (可作为 Task 或 Process 的属性，或单独实体), ReportingLine, CollaborationRelationship。
Safety_Rules_Standards (安全规章与标准)
内容: 涉及行车安全、人身安全、设备安全、食品安全、消防安全、反恐防爆等方面的具体规定、标准、禁止行为、条件要求等。
抽取侧重: SafetyRule, SafetyStandard, ProhibitedAction, Condition, Requirement, SafetyMeasure。
Emergency_Response_Plans (应急响应与预案)
内容: 针对各类突发事件（如事故、故障、自然灾害、公共卫生事件）的应急组织、预警分级、响应流程、处置措施、资源保障、报告要求等。
抽取侧重: EmergencyPlan, Event (突发事件), EventLevel, WarningLevel, ResponseProcedure, TriggerCondition, ResourceRequirement, ReportingRequirement。
Operational_Procedures_Workflows (操作流程与作业)
内容: 描述日常生产、服务、调度、维护等具体工作的操作步骤、作业标准、交接规范、记录要求、关键控制点等。
抽取侧重: Process, ProcessStep, Workflow, OperationalStandard, RecordRequirement, ControlPoint, HandoverProcedure。
Service_Quality_Standards (服务质量与规范)
内容: 涉及旅客服务标准、服务礼仪、环境卫生、服务设施配置、服务承诺、服务禁忌等方面的规定。
抽取侧重: ServiceStandard, ServiceRule, ServiceFacility, PassengerExperienceFactor, ServiceCommitment, EtiquetteRequirement。
Ticketing_Fare_Management (票务与票价管理)
内容: 涉及车票发售、预订、改签、退票、票价制定、优惠政策、实名制、票务系统操作、票据管理等规定。
抽取侧重: TicketRule, FareRule, PricingPolicy, DiscountPolicy, TicketingProcess, RealNameSystemRequirement, TicketStockManagement。
Commercial_Operations_Sales (商业运营与销售)
内容: 涉及列车餐饮、商品销售、广告、增值服务等商业经营活动的范围、规范、定价、供应商管理、合同要求等。
抽取侧重: CommercialRule, Product (商品/服务), ServiceOffering, Pricing, SupplierManagement, AdvertisingRule。
Equipment_Asset_Management (设备与资产管理)
内容: 涉及车辆、车站设施、服务设备（如餐饮设备、检票闸机）、信息系统等的配置标准、使用规范、维护保养、报废更新、资产登记等。
抽取侧重: Equipment, Asset, System, ConfigurationStandard, UsageRule, MaintenanceProcedure, LifecycleManagement。
Personnel_Management_Training (人员管理与培训)
内容: 涉及人员资质要求、岗位配置、排班、劳动纪律、仪容仪表、培训内容、考核要求、持证上岗、薪酬福利（可能少见于规章）等。
抽取侧重: PersonnelRequirement, Qualification, TrainingProgram, AssessmentStandard, WorkDiscipline, StaffingLevel, AppearanceStandard。
Supervision_Inspection_Assessment (监督检查与评估)
内容: 描述监督检查的方式、频率、内容、标准、责任主体、问题整改、考核评估方法等。
抽取侧重: InspectionProcess, AssessmentMethod, EvaluationCriteria, Frequency, ResponsibleParty, RectificationRequirement。
Incident_Complaint_Penalty_Handling (事件投诉与奖惩)
内容: 涉及服务质量问题、安全事件、旅客投诉的分类、定性、处理流程、责任认定、考核、处罚、奖励规定等。
抽取侧重: IncidentType, ComplaintType, ProblemLevel, HandlingProcedure, Accountability, Penalty, Reward, Violation。
Information_Reporting_Communication (信息报告与沟通)
内容: 涉及各类信息（运营、安全、服务、应急等）的报告路径、时限、格式、内容要求、保密要求、内部沟通协调机制等。
抽取侧重: ReportingRequirement, InformationType, ReportFormat, TimeLimit, CommunicationChannel, ConfidentialityRequirement。
Legal_Regulatory_References (法律法规引用)
内容: 主要引用或依据外部法律、法规、上级单位文件等的部分。
抽取侧重: DocumentReference (外部法规/文件), LegalBasis, ComplianceRequirement。
Supplementary_Provisions (附则与其他)
内容: 解释权归属、生效日期、废止旧规、未尽事宜处理、附件说明等。
抽取侧重: EffectiveDate, ExpiryDate, Interpreter (解释单位), AmendmentHistory, AppendixReference


```

**assistant**: 

```
Okay, I will modify the `config_simple.yaml` file based on the classifications and extraction focuses you provided.

Since the original `config_simple.yaml` is very basic, and each category requires a significantly richer set of entities and relations, I will add a new top-level section called `category_configs`. This section will contain specific schema extensions and prompt templates for each of the 15 categories you defined.

The original `schema` and `prompts` sections will remain as the "base" or "default", but the intention is that when processing a section classified under a specific category, the corresponding configuration from `category_configs` would be used (potentially merging with or overriding the base configuration depending on implementation).


# config.yaml (Modified to include category-specific schemas and prompts)
# Schema and Normalization for Corporate Regulations KG

# LightRAG 知识图谱提取配置文件

# 图谱模式定义 (Base Schema)
schema:
  # Base Entity types for LLM extraction (Core concepts)
  entity_types_llm:
    - Organization
    - Role
    - Statement       # General rule/req/def/goal/topic
    - Location        # Added as a common core type
    - System          # Added as a common core type
    - DocumentReference # Added as a common core type

  # All entity types that can exist in the graph (Base + Structural)
  all_entity_types:
    - Document
    - Section
    - Organization
    - Role
    - Statement
    - Location
    - System
    - DocumentReference
    # Note: Category-specific types will be added below

  # Mapping to Cypher labels (Base)
  entity_type_map_cypher:
    Document: Document
    Section: Section
    Organization: Organization
    Role: Role
    Statement: Statement
    Location: Location
    System: System
    DocumentReference: DocumentReference
    # Note: Category-specific mappings would need to be added dynamically or pre-defined

  # Base Relation types for LLM extraction (Core semantic links)
  relation_types_llm:
    - RESPONSIBLE_FOR # (Organization/Role -> Statement/Process/Task...)
    - BELONGS_TO      # (Organization/Role -> Organization/Role)
    - HAS_PURPOSE     # (Statement -> Statement[type=Goal])
    - REFERENCES      # (Statement/Document/Section -> Section/DocumentReference)
    - APPLIES_TO      # (Statement/Rule... -> Organization/Role/System/Location...)
    - RELATED_TO      # (Entity -> Entity) - Generic fallback
    - PART_OF         # (Organization/Role/System -> Organization/System) - Added core relation
    - CONTAINS        # (Section -> Entity) - Can be structural or semantic hint for LLM

  # All relation types that can exist in the graph (Base + Structural)
  all_relation_types:
    # Structural
    - HAS_SECTION     # (Document -> Section)
    - HAS_PARENT_SECTION # (Section -> Section)
    # Containment / Linkage
    - CONTAINS        # (Section -> Any Entity)
    # Semantic (Base)
    - RESPONSIBLE_FOR
    - BELONGS_TO
    - HAS_PURPOSE
    - REFERENCES
    - APPLIES_TO
    - PART_OF
    - RELATED_TO
    # Note: Category-specific relations will be added below

  # Mapping to Cypher relationship types (Base)
  relation_type_map_cypher:
    HAS_SECTION: HAS_SECTION
    HAS_PARENT_SECTION: HAS_PARENT_SECTION
    CONTAINS: CONTAINS
    RESPONSIBLE_FOR: RESPONSIBLE_FOR
    BELONGS_TO: BELONGS_TO
    HAS_PURPOSE: HAS_PURPOSE
    REFERENCES: REFERENCES
    APPLIES_TO: APPLIES_TO
    PART_OF: PART_OF
    RELATED_TO: RELATED_TO
    # Note: Category-specific mappings would need to be added

normalization:
  # (Keep existing, expand significantly for real-world use)
  canonical_map:
    "集团公司": "中国铁路广州局集团有限公司"
    "广铁集团": "中国铁路广州局集团有限公司"
    "客运部": "集团公司客运部" # Example, needs context or more specific rules
    # ... add many more ...

# Base Prompt 模板配置 (Can be overridden or extended by category prompts)
prompts:
  entity_extraction:
    definitions: |
      基本实体类型定义 (请在返回结果中使用括号中的英文类型名称):
      - Role (角色)：指代具有特定职责的职位或岗位，如"客运部主任"、"值班负责人"。
      - Organization (组织)：涉及的单位、部门或公司，如"中国铁路广州局集团有限公司"、"集团公司客运部"。
      - Statement (陈述)：通用的规则、要求、定义、标准、目标、条件或规定。
      - Location (地点): 物理位置或区域，如"广州南站"、"京广线"。
      - System (系统): 信息系统、设备系统等，如"客票发售和预订系统"。
      - DocumentReference (文档引用): 对其他文档、法规、标准的引用，如"《安全管理条例》"。
      # 注意：具体分类的Prompt会增加更详细的类型定义。
    template: |
      请从以下文本中提取实体。

      {definitions} # Definitions will be replaced/augmented by category-specific prompts

      **要提取的实体类型列表**: {entity_types}

      文档信息 (仅供参考):
      文档标题: {document_title}
      当前章节: {current_heading}
      章节路径: {section_path}

      文本：
      """
      {content}
      """

      【重要】：必须使用英文实体类型！返回严格的 JSON 格式，包含 "entities" 列表。
      输出格式示例:
      {
        "entities": [
          {"name": "集团公司客运部", "type": "Organization"},
          {"name": "站务员", "type": "Role"}
        ]
      }
  relation_extraction:
    definitions: |
      基本关系类型定义 (请在返回结果中使用括号中的英文关系类型):
      - BELONGS_TO (隶属关系): 指明组织或角色的归属。 (Organization/Role -> Organization/Role)
      - RESPONSIBLE_FOR (责任): 明确组织或角色对某项事务的职责。 (Organization/Role -> Statement/Process/Task...)
      - REFERENCES (引用): 明确引用某章节或文档。 (Statement/Document/Section -> Section/DocumentReference)
      - APPLIES_TO (适用于): 规则/陈述适用的对象。 (Statement/Rule... -> Organization/Role/System/Location...)
      - PART_OF (一部分): 表示从属或组成关系。 (Organization/Role/System -> Organization/System)
      - RELATED_TO (关联): 连接实体间的其他关联。 (Entity -> Entity)
      # 注意：具体分类的Prompt会增加更详细的关系定义。
    template: |
      请从以下文本中，根据预定义的实体列表，提取这些实体之间符合定义的关系类型。

      {definitions} # Definitions will be replaced/augmented by category-specific prompts

      **要提取的关系类型列表**: {relation_types}
      预定义的实体类型列表 (用于关系端点): {all_entity_types_english}

      文档信息 (仅供参考):
      文档标题: {document_title}
      当前章节: {current_heading}

      文本：
      """
      {content}
      """

      文本中已识别的实体列表:
      """
      {entities_json}
      """

      【重要】：必须使用英文实体类型和关系类型！返回严格的 JSON 格式，包含 "relations" 列表。
      输出格式示例:
      {
        "relations": [
          {
            "source": "集团公司客运部", "source_type": "Organization",
            "target": "多元经营管理处", "target_type": "Organization",
            "type": "PART_OF"
          },
          {
            "source": "站务员", "source_type": "Role",
            "target": "旅客引导", "target_type": "Task", # Assuming Task is defined in category
            "type": "RESPONSIBLE_FOR"
          }
        ]
      }

# =============================================
# Category-Specific Schemas and Prompts
# =============================================
category_configs:

  General_Principles_Definitions:
    schema_extension:
      entity_types:
        - Purpose
        - Scope # Can be represented by APPLIES_TO relation target types
        - GuidingPrinciple
        - Definition # Often captured as Statement, but could be specific type
      relation_types:
        - DEFINES # (Section/Document -> Purpose/Definition/GuidingPrinciple)
        # APPLIES_TO relation (base) covers Scope well
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - Purpose (目的): 规章的核心目标或意图。
          - GuidingPrinciple (指导原则): 规章遵循的基本原则。
          - Definition (定义): 对特定术语的解释。
          - Organization (组织): 涉及的单位、部门。
          - Location (地点): 适用的地理范围。
          - System (系统): 适用的系统。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注规章的制定目的、适用范围（组织、地点、系统）、基本原则和术语定义**。

          {definitions}
          **要提取的实体类型列表**: [Purpose, GuidingPrinciple, Definition, Organization, Location, System, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - DEFINES (定义): 章节/文档定义了目的/原则/术语。 (Section/Document -> Purpose/GuidingPrinciple/Definition)
          - APPLIES_TO (适用于): 规章适用于哪些组织/地点/系统。 (Document/Section/Statement -> Organization/Location/System)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注规章定义了什么 (DEFINES)，以及它适用于哪些对象 (APPLIES_TO)**。

          {definitions}
          **要提取的关系类型列表**: [DEFINES, APPLIES_TO, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Organizational_Responsibilities:
    schema_extension:
      entity_types:
        - Responsibility # Can be Statement or specific Task/Process type
        - ReportingLine # Often represented by BELONGS_TO or specific relation
        - CollaborationRelationship # Often represented by RELATED_TO or specific relation
        - Task # Added for specific responsibilities
        - Process # Added for specific responsibilities
      relation_types:
        - HAS_RESPONSIBILITY # (Organization/Role -> Responsibility/Task/Process) - More specific than RESPONSIBLE_FOR
        - REPORTS_TO # (Role/Organization -> Role/Organization) - Specific reporting line
        - COLLABORATES_WITH # (Organization/Role -> Organization/Role)
        # RESPONSIBLE_FOR (base) is key here
        # BELONGS_TO / PART_OF (base) are key here
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - Organization (组织): 涉及的单位、部门。
          - Role (角色): 职位或岗位。
          - Responsibility (职责): 描述性的职责陈述 (可用Statement)。
          - Task (任务): 具体的、可操作的任务职责。
          - Process (流程): 涉及多个步骤的职责流程。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注涉及的组织、角色/岗位，以及他们承担的具体职责、任务或流程**。

          {definitions}
          **要提取的实体类型列表**: [Organization, Role, Responsibility, Task, Process, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - RESPONSIBLE_FOR (责任): 组织/角色负责某项职责/任务/流程。 (Organization/Role -> Responsibility/Task/Process/Statement)
          - REPORTS_TO (汇报): 明确的汇报关系。 (Role/Organization -> Role/Organization)
          - COLLABORATES_WITH (协作): 明确的协作关系。 (Organization/Role -> Organization/Role)
          - BELONGS_TO / PART_OF (隶属/组成): 组织/角色的层级关系。 (Organization/Role -> Organization/Role)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注谁负责什么 (RESPONSIBLE_FOR)，组织/角色间的层级/汇报关系 (BELONGS_TO/PART_OF/REPORTS_TO)，以及协作关系 (COLLABORATES_WITH)**。

          {definitions}
          **要提取的关系类型列表**: [RESPONSIBLE_FOR, REPORTS_TO, COLLABORATES_WITH, BELONGS_TO, PART_OF, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Safety_Rules_Standards:
    schema_extension:
      entity_types:
        - SafetyRule
        - SafetyStandard
        - ProhibitedAction
        - Condition
        - Requirement # Often Statement or specific type
        - SafetyMeasure
        - Equipment # Added as relevant context
      relation_types:
        - HAS_STANDARD # (SafetyRule/Process/Equipment -> SafetyStandard)
        - REQUIRES # (SafetyRule/Process -> Requirement/Condition/SafetyMeasure/Equipment)
        - PROHIBITS # (SafetyRule -> ProhibitedAction/Action)
        - MUST_COMPLY_WITH # (Organization/Role/Equipment/Process -> SafetyRule/SafetyStandard)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - SafetyRule (安全规则): 具体的安全规定条款。
          - SafetyStandard (安全标准): 量化的或定性的安全标准。
          - ProhibitedAction (禁止行为): 明确禁止的操作或行为。
          - Condition (条件): 规则/标准适用的条件。
          - Requirement (要求): 必须满足的要求 (可用Statement)。
          - SafetyMeasure (安全措施): 为保障安全采取的具体措施。
          - Equipment (设备): 涉及安全的设备。
          # ... (Include relevant base types like Org, Role) ...
        template: |
          请从以下文本中提取实体，**重点关注具体的安全规则、安全标准、禁止行为、条件、要求、安全措施以及涉及的设备**。

          {definitions}
          **要提取的实体类型列表**: [SafetyRule, SafetyStandard, ProhibitedAction, Condition, Requirement, SafetyMeasure, Equipment, Organization, Role, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - HAS_STANDARD (有标准): 规则/流程/设备关联的安全标准。 (SafetyRule/Process/Equipment -> SafetyStandard)
          - REQUIRES (要求): 规则/流程需要满足的要求/条件/措施/设备。 (SafetyRule/Process -> Requirement/Condition/SafetyMeasure/Equipment)
          - PROHIBITS (禁止): 规则禁止的行为。 (SafetyRule -> ProhibitedAction)
          - MUST_COMPLY_WITH (须遵守): 对象必须遵守的规则/标准。 (Organization/Role/Equipment/Process -> SafetyRule/SafetyStandard)
          - APPLIES_TO (适用于): 规则/标准适用的对象。 (SafetyRule/SafetyStandard -> Organization/Role/Equipment/Location...)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注规则/标准的内容 (HAS_STANDARD, REQUIRES, PROHIBITS)，以及谁/什么需要遵守这些规则/标准 (MUST_COMPLY_WITH, APPLIES_TO)**。

          {definitions}
          **要提取的关系类型列表**: [HAS_STANDARD, REQUIRES, PROHIBITS, MUST_COMPLY_WITH, APPLIES_TO, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Emergency_Response_Plans:
    schema_extension:
      entity_types:
        - EmergencyPlan
        - Event # Specific type for incidents/emergencies
        - EventLevel
        - WarningLevel
        - ResponseProcedure # Specific Process type
        - TriggerCondition # Specific Condition type
        - ResourceRequirement # Specific Requirement type
        - ReportingRequirement # Specific Requirement type
        - Resource # Added for clarity
      relation_types:
        - TRIGGERS # (Event/EventLevel/WarningLevel/TriggerCondition -> ResponseProcedure/EmergencyPlan)
        - HAS_PROCEDURE # (EmergencyPlan -> ResponseProcedure)
        - HAS_LEVEL # (Event/Warning -> EventLevel/WarningLevel)
        - REQUIRES_RESOURCE # (ResponseProcedure/EmergencyPlan -> Resource/ResourceRequirement)
        - REQUIRES_REPORTING # (ResponseProcedure/Event -> ReportingRequirement)
        - ACTIVATED_BY # (ResponseProcedure/EmergencyPlan -> Organization/Role)
        # RESPONSIBLE_FOR (base) is key for roles in the plan
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - EmergencyPlan (应急预案): 整体应急计划。
          - Event (突发事件): 可能触发预案的事件。
          - EventLevel (事件等级): 事件的严重程度分级。
          - WarningLevel (预警等级): 预警信号的级别。
          - ResponseProcedure (响应流程): 应急处置的具体流程/步骤。
          - TriggerCondition (触发条件): 启动预案或流程的具体条件。
          - ResourceRequirement (资源需求): 应急所需的资源说明。
          - ReportingRequirement (报告要求): 应急相关的报告规定。
          - Resource (资源): 具体的人力、物力资源。
          - Organization (应急组织): 参与应急的单位。
          - Role (应急角色): 参与应急的岗位。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注应急预案本身、可能发生的事件、事件/预警分级、响应流程、触发条件、资源和报告要求，以及相关的应急组织和角色**。

          {definitions}
          **要提取的实体类型列表**: [EmergencyPlan, Event, EventLevel, WarningLevel, ResponseProcedure, TriggerCondition, ResourceRequirement, ReportingRequirement, Resource, Organization, Role, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - TRIGGERS (触发): 事件/条件触发预案/流程。 (Event/EventLevel/WarningLevel/TriggerCondition -> ResponseProcedure/EmergencyPlan)
          - HAS_PROCEDURE (包含流程): 预案包含的响应流程。 (EmergencyPlan -> ResponseProcedure)
          - HAS_LEVEL (有等级): 事件/预警对应的等级。 (Event/Warning -> EventLevel/WarningLevel)
          - REQUIRES_RESOURCE (需要资源): 流程/预案需要的资源。 (ResponseProcedure/EmergencyPlan -> Resource/ResourceRequirement)
          - REQUIRES_REPORTING (需要报告): 流程/事件的报告要求。 (ResponseProcedure/Event -> ReportingRequirement)
          - ACTIVATED_BY (由...启动): 预案/流程由谁启动。 (ResponseProcedure/EmergencyPlan -> Organization/Role)
          - RESPONSIBLE_FOR (负责): 组织/角色在预案/流程中的职责。 (Organization/Role -> ResponseProcedure/Task)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注什么触发了预案/流程 (TRIGGERS)，预案包含哪些流程 (HAS_PROCEDURE)，流程需要什么资源/报告 (REQUIRES_RESOURCE/REQUIRES_REPORTING)，以及由谁负责/启动 (RESPONSIBLE_FOR/ACTIVATED_BY)**。

          {definitions}
          **要提取的关系类型列表**: [TRIGGERS, HAS_PROCEDURE, HAS_LEVEL, REQUIRES_RESOURCE, REQUIRES_REPORTING, ACTIVATED_BY, RESPONSIBLE_FOR, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Operational_Procedures_Workflows:
    schema_extension:
      entity_types:
        - Process
        - ProcessStep
        - Workflow # Can be alias for Process or distinct
        - OperationalStandard
        - RecordRequirement
        - ControlPoint
        - HandoverProcedure # Specific ProcessStep or Process type
        - InputData # Added for clarity
        - OutputData # Added for clarity
      relation_types:
        - HAS_STEP # (Process/Workflow -> ProcessStep)
        - FOLLOWS # (ProcessStep -> ProcessStep) - Sequence
        - REQUIRES_INPUT # (Process/ProcessStep -> InputData/Resource/System)
        - PRODUCES_OUTPUT # (Process/ProcessStep -> OutputData/RecordRequirement)
        - HAS_STANDARD # (Process/ProcessStep -> OperationalStandard)
        - IS_CONTROL_POINT # (ProcessStep -> ControlPoint) - Or ControlPoint as boolean property
        - INVOLVES_HANDOVER # (Process/ProcessStep -> HandoverProcedure)
        # RESPONSIBLE_FOR (base) is key
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - Process (流程): 日常操作或作业流程。
          - ProcessStep (流程步骤): 流程中的具体步骤。
          - Workflow (工作流): 通常指自动化或系统化的流程。
          - OperationalStandard (作业标准): 操作必须达到的标准。
          - RecordRequirement (记录要求): 需要记录的信息或文档。
          - ControlPoint (控制点): 流程中的关键检查或控制环节。
          - HandoverProcedure (交接程序): 涉及工作交接的步骤或规定。
          - InputData (输入数据): 流程/步骤需要的输入。
          - OutputData (输出数据): 流程/步骤产生的输出。
          # ... (Include relevant base types like Org, Role, System) ...
        template: |
          请从以下文本中提取实体，**重点关注具体的操作流程、流程步骤、作业标准、记录要求、关键控制点、交接程序以及输入输出**。

          {definitions}
          **要提取的实体类型列表**: [Process, ProcessStep, Workflow, OperationalStandard, RecordRequirement, ControlPoint, HandoverProcedure, InputData, OutputData, Organization, Role, System, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - HAS_STEP (有步骤): 流程包含的步骤。 (Process/Workflow -> ProcessStep)
          - FOLLOWS (顺序): 步骤之间的先后顺序。 (ProcessStep -> ProcessStep)
          - REQUIRES_INPUT (需要输入): 流程/步骤需要的输入。 (Process/ProcessStep -> InputData/Resource/System)
          - PRODUCES_OUTPUT (产生输出): 流程/步骤产生的输出/记录。 (Process/ProcessStep -> OutputData/RecordRequirement)
          - HAS_STANDARD (有标准): 流程/步骤关联的作业标准。 (Process/ProcessStep -> OperationalStandard)
          - IS_CONTROL_POINT (是控制点): 标识步骤为关键控制点。 (ProcessStep -> ControlPoint)
          - INVOLVES_HANDOVER (涉及交接): 流程/步骤涉及交接。 (Process/ProcessStep -> HandoverProcedure)
          - RESPONSIBLE_FOR (负责): 谁负责执行流程/步骤。 (Organization/Role -> Process/ProcessStep)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注流程的步骤顺序 (HAS_STEP, FOLLOWS)，步骤的输入输出 (REQUIRES_INPUT, PRODUCES_OUTPUT)，关联的标准 (HAS_STANDARD)，以及谁负责执行 (RESPONSIBLE_FOR)**。

          {definitions}
          **要提取的关系类型列表**: [HAS_STEP, FOLLOWS, REQUIRES_INPUT, PRODUCES_OUTPUT, HAS_STANDARD, IS_CONTROL_POINT, INVOLVES_HANDOVER, RESPONSIBLE_FOR, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Service_Quality_Standards:
    schema_extension:
      entity_types:
        - ServiceStandard
        - ServiceRule
        - ServiceFacility # Specific Equipment type
        - PassengerExperienceFactor # Can be Statement or specific type
        - ServiceCommitment
        - EtiquetteRequirement
        - ServiceScenario # Added context
      relation_types:
        - DEFINES_STANDARD # (Section/Document -> ServiceStandard/ServiceRule...)
        - APPLIES_TO_SERVICE # (ServiceStandard/ServiceRule... -> ServiceScenario/Role/Location/ServiceFacility)
        - REQUIRES_FACILITY # (ServiceStandard/ServiceScenario -> ServiceFacility)
        - IMPACTS_EXPERIENCE # (ServiceStandard/ServiceRule/ServiceFacility -> PassengerExperienceFactor)
        - COMMITS_TO # (Organization/Document -> ServiceCommitment)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - ServiceStandard (服务标准): 对服务质量的具体量化或定性标准。
          - ServiceRule (服务规则): 服务过程中必须遵守的规则。
          - ServiceFacility (服务设施): 提供服务所需的设施设备。
          - PassengerExperienceFactor (旅客体验因素): 影响旅客感受的关键点。
          - ServiceCommitment (服务承诺): 对外公开的服务保证。
          - EtiquetteRequirement (礼仪要求): 对服务人员行为举止的要求。
          - ServiceScenario (服务场景): 规则/标准适用的具体服务环节。
          # ... (Include relevant base types like Org, Role, Location) ...
        template: |
          请从以下文本中提取实体，**重点关注服务标准、服务规则、服务设施、影响旅客体验的因素、服务承诺以及礼仪要求**。

          {definitions}
          **要提取的实体类型列表**: [ServiceStandard, ServiceRule, ServiceFacility, PassengerExperienceFactor, ServiceCommitment, EtiquetteRequirement, ServiceScenario, Organization, Role, Location, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - DEFINES_STANDARD (定义标准): 定义了服务标准/规则。 (Section/Document -> ServiceStandard/ServiceRule...)
          - APPLIES_TO_SERVICE (适用于服务): 标准/规则适用于哪个场景/角色/地点/设施。 (ServiceStandard/ServiceRule... -> ServiceScenario/Role/Location/ServiceFacility)
          - REQUIRES_FACILITY (需要设施): 服务标准/场景需要的设施。 (ServiceStandard/ServiceScenario -> ServiceFacility)
          - IMPACTS_EXPERIENCE (影响体验): 标准/规则/设施如何影响旅客体验。 (ServiceStandard/ServiceRule/ServiceFacility -> PassengerExperienceFactor)
          - COMMITS_TO (承诺): 组织/文档做出的服务承诺。 (Organization/Document -> ServiceCommitment)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注标准/规则适用于哪些服务场景/对象 (APPLIES_TO_SERVICE)，需要哪些设施 (REQUIRES_FACILITY)，如何影响旅客体验 (IMPACTS_EXPERIENCE)，以及服务承诺 (COMMITS_TO)**。

          {definitions}
          **要提取的关系类型列表**: [DEFINES_STANDARD, APPLIES_TO_SERVICE, REQUIRES_FACILITY, IMPACTS_EXPERIENCE, COMMITS_TO, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Ticketing_Fare_Management:
    schema_extension:
      entity_types:
        - TicketRule
        - FareRule
        - PricingPolicy
        - DiscountPolicy
        - TicketingProcess # Specific Process type
        - RealNameSystemRequirement # Specific Requirement type
        - TicketStockManagement # Specific Process/Rule type
        - TicketType # Added context
        - FareZone # Added context
        - PaymentMethod # Added context
      relation_types:
        - GOVERNS_TICKET # (TicketRule/FareRule/PricingPolicy/DiscountPolicy -> TicketType/FareZone/Condition)
        - PART_OF_PROCESS # (TicketRule/FareRule... -> TicketingProcess)
        - REQUIRES_REALNAME # (TicketingProcess/TicketType -> RealNameSystemRequirement)
        - MANAGES_STOCK # (TicketStockManagement -> TicketType)
        - ACCEPTS_PAYMENT # (TicketingProcess -> PaymentMethod)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - TicketRule (票务规则): 售票、改签、退票等具体规则。
          - FareRule (票价规则): 票价计算、应用规则。
          - PricingPolicy (定价策略): 票价制定的原则或方法。
          - DiscountPolicy (优惠政策): 票价优惠的具体规定。
          - TicketingProcess (票务流程): 购票、检票等流程。
          - RealNameSystemRequirement (实名制要求): 实名制相关规定。
          - TicketStockManagement (票据管理): 车票库存、票据相关的管理规定。
          - TicketType (票种): 如成人票、学生票。
          - FareZone (计价区段): 如特定线路或里程。
          - PaymentMethod (支付方式): 如现金、支付宝。
          # ... (Include relevant base types like Org, Role, System) ...
        template: |
          请从以下文本中提取实体，**重点关注票务规则（售、改、退）、票价规则、定价策略、优惠政策、票务流程、实名制要求和票据管理规定**。

          {definitions}
          **要提取的实体类型列表**: [TicketRule, FareRule, PricingPolicy, DiscountPolicy, TicketingProcess, RealNameSystemRequirement, TicketStockManagement, TicketType, FareZone, PaymentMethod, Organization, Role, System, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - GOVERNS_TICKET (管理票务): 规则/政策适用于哪种票/区段/条件。 (TicketRule/FareRule/PricingPolicy/DiscountPolicy -> TicketType/FareZone/Condition)
          - PART_OF_PROCESS (属于流程): 规则是哪个票务流程的一部分。 (TicketRule/FareRule... -> TicketingProcess)
          - REQUIRES_REALNAME (要求实名): 流程/票种的实名制要求。 (TicketingProcess/TicketType -> RealNameSystemRequirement)
          - MANAGES_STOCK (管理票据): 票据管理规定涉及的票种。 (TicketStockManagement -> TicketType)
          - ACCEPTS_PAYMENT (接受支付): 票务流程接受的支付方式。 (TicketingProcess -> PaymentMethod)
          # ... (Include relevant base relations like RESPONSIBLE_FOR) ...
        template: |
          请提取实体间的关系，**重点关注各项规则/政策适用于哪些票种/场景 (GOVERNS_TICKET)，它们属于哪个流程 (PART_OF_PROCESS)，以及实名制要求 (REQUIRES_REALNAME) 和支付方式 (ACCEPTS_PAYMENT)**。

          {definitions}
          **要提取的关系类型列表**: [GOVERNS_TICKET, PART_OF_PROCESS, REQUIRES_REALNAME, MANAGES_STOCK, ACCEPTS_PAYMENT, RESPONSIBLE_FOR, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Commercial_Operations_Sales:
    schema_extension:
      entity_types:
        - CommercialRule
        - Product # Goods or services offered
        - ServiceOffering # Specific service package/offering
        - Pricing # Specific price or pricing model
        - SupplierManagement # Process/Rule for suppliers
        - AdvertisingRule
        - Supplier # Specific Organization type
        - Contract # Added context
      relation_types:
        - GOVERNS_COMMERCIAL # (CommercialRule/AdvertisingRule -> Product/ServiceOffering/Location/Organization)
        - OFFERS # (Organization/Location -> Product/ServiceOffering)
        - HAS_PRICE # (Product/ServiceOffering -> Pricing)
        - MANAGES_SUPPLIER # (SupplierManagement -> Supplier/Contract)
        - SUPPLIED_BY # (Product/ServiceOffering -> Supplier)
        - HAS_CONTRACT_WITH # (Organization -> Supplier/Contract)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - CommercialRule (商业规则): 商业经营活动规范。
          - Product (商品/服务): 列车上销售的商品或提供的服务。
          - ServiceOffering (服务项目): 具体打包的服务内容。
          - Pricing (定价): 商品/服务的价格信息或定价规则。
          - SupplierManagement (供应商管理): 对供应商的管理规定。
          - AdvertisingRule (广告规则): 商业广告相关规定。
          - Supplier (供应商): 提供商品/服务的外部单位。
          - Contract (合同): 与供应商等的合同。
          # ... (Include relevant base types like Org, Role, Location) ...
        template: |
          请从以下文本中提取实体，**重点关注商业经营规则、提供的商品/服务、定价、供应商管理规定、广告规则、供应商以及相关合同**。

          {definitions}
          **要提取的实体类型列表**: [CommercialRule, Product, ServiceOffering, Pricing, SupplierManagement, AdvertisingRule, Supplier, Contract, Organization, Role, Location, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - GOVERNS_COMMERCIAL (管理商业): 规则适用于哪个商品/服务/地点/组织。 (CommercialRule/AdvertisingRule -> Product/ServiceOffering/Location/Organization)
          - OFFERS (提供): 谁/在哪里提供商品/服务。 (Organization/Location -> Product/ServiceOffering)
          - HAS_PRICE (有价格): 商品/服务的定价。 (Product/ServiceOffering -> Pricing)
          - MANAGES_SUPPLIER (管理供应商): 供应商管理规定涉及的供应商/合同。 (SupplierManagement -> Supplier/Contract)
          - SUPPLIED_BY (由...供应): 商品/服务由哪个供应商提供。 (Product/ServiceOffering -> Supplier)
          - HAS_CONTRACT_WITH (有合同): 与供应商的合同关系。 (Organization -> Supplier/Contract)
          # ... (Include relevant base relations like RESPONSIBLE_FOR) ...
        template: |
          请提取实体间的关系，**重点关注商业规则适用于什么 (GOVERNS_COMMERCIAL)，谁提供什么商品/服务 (OFFERS)，价格是多少 (HAS_PRICE)，供应商如何管理 (MANAGES_SUPPLIER, SUPPLIED_BY, HAS_CONTRACT_WITH)**。

          {definitions}
          **要提取的关系类型列表**: [GOVERNS_COMMERCIAL, OFFERS, HAS_PRICE, MANAGES_SUPPLIER, SUPPLIED_BY, HAS_CONTRACT_WITH, RESPONSIBLE_FOR, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Equipment_Asset_Management:
    schema_extension:
      entity_types:
        - Equipment
        - Asset # Broader than Equipment
        - System # Re-emphasized here
        - ConfigurationStandard
        - UsageRule
        - MaintenanceProcedure # Specific Process type
        - LifecycleManagement # Process/Rule for asset lifecycle
        - AssetStatus # Added context (e.g., 在用, 报废)
      relation_types:
        - HAS_STANDARD # (Equipment/Asset/System -> ConfigurationStandard)
        - GOVERNED_BY_RULE # (Equipment/Asset/System -> UsageRule)
        - REQUIRES_MAINTENANCE # (Equipment/Asset/System -> MaintenanceProcedure)
        - HAS_LIFECYCLE_POLICY # (Equipment/Asset/System -> LifecycleManagement)
        - HAS_STATUS # (Asset/Equipment -> AssetStatus)
        - INSTALLED_AT # (Equipment/Asset/System -> Location)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - Equipment (设备): 具体的操作设备，如车辆、闸机。
          - Asset (资产): 更广义的资产概念。
          - System (系统): 信息系统或设备系统。
          - ConfigurationStandard (配置标准): 设备/资产/系统的配置要求。
          - UsageRule (使用规范): 设备/资产/系统的使用规定。
          - MaintenanceProcedure (维护保养规程): 维护的具体流程。
          - LifecycleManagement (生命周期管理): 资产从配置到报废的管理规定。
          - AssetStatus (资产状态): 如在用、维修中、报废。
          # ... (Include relevant base types like Org, Role, Location) ...
        template: |
          请从以下文本中提取实体，**重点关注具体的设备、资产、系统，以及它们的配置标准、使用规范、维护保养规程和生命周期管理规定**。

          {definitions}
          **要提取的实体类型列表**: [Equipment, Asset, System, ConfigurationStandard, UsageRule, MaintenanceProcedure, LifecycleManagement, AssetStatus, Organization, Role, Location, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - HAS_STANDARD (有配置标准): 设备/资产/系统的配置标准。 (Equipment/Asset/System -> ConfigurationStandard)
          - GOVERNED_BY_RULE (受规则约束): 设备/资产/系统的使用规则。 (Equipment/Asset/System -> UsageRule)
          - REQUIRES_MAINTENANCE (需维护): 设备/资产/系统需要的维护规程。 (Equipment/Asset/System -> MaintenanceProcedure)
          - HAS_LIFECYCLE_POLICY (有生命周期策略): 设备/资产/系统的生命周期管理策略。 (Equipment/Asset/System -> LifecycleManagement)
          - HAS_STATUS (状态为): 资产/设备的状态。 (Asset/Equipment -> AssetStatus)
          - INSTALLED_AT (安装于): 设备/资产/系统的位置。 (Equipment/Asset/System -> Location)
          - RESPONSIBLE_FOR (负责): 谁负责管理/维护设备/资产。 (Organization/Role -> Equipment/Asset/MaintenanceProcedure)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注设备/资产/系统的配置标准 (HAS_STANDARD)、使用规则 (GOVERNED_BY_RULE)、维护要求 (REQUIRES_MAINTENANCE)、生命周期管理 (HAS_LIFECYCLE_POLICY) 以及由谁负责 (RESPONSIBLE_FOR)**。

          {definitions}
          **要提取的关系类型列表**: [HAS_STANDARD, GOVERNED_BY_RULE, REQUIRES_MAINTENANCE, HAS_LIFECYCLE_POLICY, HAS_STATUS, INSTALLED_AT, RESPONSIBLE_FOR, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Personnel_Management_Training:
    schema_extension:
      entity_types:
        - PersonnelRequirement # Specific Requirement type
        - Qualification
        - TrainingProgram # Specific Process type
        - AssessmentStandard # Specific Standard type
        - WorkDiscipline # Specific Rule type
        - StaffingLevel # Specific Requirement type
        - AppearanceStandard # Specific Standard type
        - Position # Alias for Role or distinct
        - Certification # Added context
      relation_types:
        - REQUIRES_QUALIFICATION # (Role/Position/Task -> Qualification/Certification)
        - HAS_REQUIREMENT # (Role/Position -> PersonnelRequirement/StaffingLevel)
        - INCLUDES_TRAINING # (Organization/Role -> TrainingProgram)
        - ASSESSED_BY # (Role/Qualification/TrainingProgram -> AssessmentStandard)
        - MUST_ADHERE_TO # (Role/Organization -> WorkDiscipline/AppearanceStandard)
        - REQUIRES_CERTIFICATION # (Role/Position -> Certification)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - PersonnelRequirement (人员要求): 对人员的基本要求。
          - Qualification (资质): 上岗所需的资格或能力。
          - TrainingProgram (培训项目): 具体的培训内容或计划。
          - AssessmentStandard (考核标准): 人员考核的标准。
          - WorkDiscipline (劳动纪律): 工作中需遵守的纪律规定。
          - StaffingLevel (人员配置): 岗位人员数量要求。
          - AppearanceStandard (仪容仪表标准): 对员工外观的要求。
          - Position (岗位): 具体的工作岗位 (可用Role)。
          - Certification (证书): 上岗所需的证书。
          # ... (Include relevant base types like Org, Role) ...
        template: |
          请从以下文本中提取实体，**重点关注人员资质要求、岗位配置、培训内容、考核标准、劳动纪律、仪容仪表标准以及持证上岗要求**。

          {definitions}
          **要提取的实体类型列表**: [PersonnelRequirement, Qualification, TrainingProgram, AssessmentStandard, WorkDiscipline, StaffingLevel, AppearanceStandard, Position, Certification, Organization, Role, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - REQUIRES_QUALIFICATION (要求资质): 岗位/任务需要的资质/证书。 (Role/Position/Task -> Qualification/Certification)
          - HAS_REQUIREMENT (有要求): 岗位的人员/配置要求。 (Role/Position -> PersonnelRequirement/StaffingLevel)
          - INCLUDES_TRAINING (包含培训): 组织/角色涉及的培训项目。 (Organization/Role -> TrainingProgram)
          - ASSESSED_BY (通过...考核): 角色/资质/培训通过何种标准考核。 (Role/Qualification/TrainingProgram -> AssessmentStandard)
          - MUST_ADHERE_TO (须遵守): 角色/组织需遵守的纪律/标准。 (Role/Organization -> WorkDiscipline/AppearanceStandard)
          - REQUIRES_CERTIFICATION (要求证书): 岗位需要的证书。 (Role/Position -> Certification)
          # ... (Include relevant base relations like RESPONSIBLE_FOR training) ...
        template: |
          请提取实体间的关系，**重点关注岗位需要什么资质/证书 (REQUIRES_QUALIFICATION/REQUIRES_CERTIFICATION)，涉及哪些培训 (INCLUDES_TRAINING)，考核标准是什么 (ASSESSED_BY)，以及需要遵守的纪律/规范 (MUST_ADHERE_TO)**。

          {definitions}
          **要提取的关系类型列表**: [REQUIRES_QUALIFICATION, HAS_REQUIREMENT, INCLUDES_TRAINING, ASSESSED_BY, MUST_ADHERE_TO, REQUIRES_CERTIFICATION, RESPONSIBLE_FOR, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Supervision_Inspection_Assessment:
    schema_extension:
      entity_types:
        - InspectionProcess # Specific Process type
        - AssessmentMethod
        - EvaluationCriteria # Specific Standard/Requirement type
        - Frequency # Can be property or specific type
        - ResponsibleParty # Use Organization/Role
        - RectificationRequirement # Specific Requirement type
        - InspectionContent # Added context
        - InspectionScope # Added context
      relation_types:
        - USES_METHOD # (InspectionProcess -> AssessmentMethod)
        - HAS_CRITERIA # (InspectionProcess/AssessmentMethod -> EvaluationCriteria)
        - HAS_FREQUENCY # (InspectionProcess -> Frequency)
        - CONDUCTED_BY # (InspectionProcess -> ResponsibleParty/Organization/Role)
        - REQUIRES_RECTIFICATION # (InspectionProcess/Violation -> RectificationRequirement)
        - COVERS_CONTENT # (InspectionProcess -> InspectionContent)
        - APPLIES_TO_SCOPE # (InspectionProcess -> InspectionScope/Organization/Location...)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - InspectionProcess (监督检查流程): 检查活动的过程。
          - AssessmentMethod (评估方法): 采用的具体评估手段。
          - EvaluationCriteria (评估标准): 判断好坏的标准。
          - Frequency (频率): 检查/评估的频次。
          - ResponsibleParty (责任方): 执行检查/评估的单位或角色 (用Organization/Role)。
          - RectificationRequirement (整改要求): 发现问题后的整改规定。
          - InspectionContent (检查内容): 检查的具体项目。
          - InspectionScope (检查范围): 检查的对象范围。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注监督检查的方式、频率、内容、标准、责任主体以及问题整改要求**。

          {definitions}
          **要提取的实体类型列表**: [InspectionProcess, AssessmentMethod, EvaluationCriteria, Frequency, Organization, Role, RectificationRequirement, InspectionContent, InspectionScope, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - USES_METHOD (使用方法): 检查流程使用的方法。 (InspectionProcess -> AssessmentMethod)
          - HAS_CRITERIA (有标准): 检查/评估的标准。 (InspectionProcess/AssessmentMethod -> EvaluationCriteria)
          - HAS_FREQUENCY (有频率): 检查流程的频率。 (InspectionProcess -> Frequency)
          - CONDUCTED_BY (由...执行): 检查由谁负责。 (InspectionProcess -> Organization/Role)
          - REQUIRES_RECTIFICATION (要求整改): 检查/违规引发的整改要求。 (InspectionProcess/Violation -> RectificationRequirement)
          - COVERS_CONTENT (覆盖内容): 检查流程覆盖的内容。 (InspectionProcess -> InspectionContent)
          - APPLIES_TO_SCOPE (适用于范围): 检查流程适用的范围。 (InspectionProcess -> InspectionScope/Organization/Location...)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注检查使用的方法 (USES_METHOD)、标准 (HAS_CRITERIA)、频率 (HAS_FREQUENCY)、由谁执行 (CONDUCTED_BY) 以及整改要求 (REQUIRES_RECTIFICATION)**。

          {definitions}
          **要提取的关系类型列表**: [USES_METHOD, HAS_CRITERIA, HAS_FREQUENCY, CONDUCTED_BY, REQUIRES_RECTIFICATION, COVERS_CONTENT, APPLIES_TO_SCOPE, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Incident_Complaint_Penalty_Handling:
    schema_extension:
      entity_types:
        - IncidentType
        - ComplaintType
        - ProblemLevel
        - HandlingProcedure # Specific Process type
        - Accountability # Often Role/Organization + RESPONSIBLE_FOR
        - Penalty
        - Reward
        - Violation # Specific Event/Action type
        - ComplaintSource # Added context
      relation_types:
        - CLASSIFIED_AS # (Violation/Event/Complaint -> IncidentType/ComplaintType/ProblemLevel)
        - HANDLED_BY_PROCEDURE # (IncidentType/ComplaintType/ProblemLevel/Violation -> HandlingProcedure)
        - LEADS_TO_PENALTY # (Violation/ProblemLevel -> Penalty)
        - LEADS_TO_REWARD # (Action/Performance -> Reward)
        - DETERMINES_ACCOUNTABILITY # (HandlingProcedure/Investigation -> Accountability/Organization/Role)
        - APPLIES_PENALTY # (Organization/Role -> Penalty -> Organization/Role/Violation) # Ternary or simplified
        - RECEIVES_COMPLAINT_FROM # (HandlingProcedure/Organization -> ComplaintSource)
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - IncidentType (事件类型): 如安全事件、服务质量事件。
          - ComplaintType (投诉类型): 如票务投诉、环境投诉。
          - ProblemLevel (问题级别): 如一般问题、严重问题。
          - HandlingProcedure (处理流程): 事件/投诉的处理步骤。
          - Accountability (责任认定): 涉及的责任方 (用Organization/Role)。
          - Penalty (处罚): 具体的处罚措施。
          - Reward (奖励): 具体的奖励措施。
          - Violation (违规行为): 具体违反规定的行为。
          - ComplaintSource (投诉来源): 如旅客、内部提报。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注事件/投诉的类型、问题级别、处理流程、责任认定、处罚和奖励规定以及具体的违规行为**。

          {definitions}
          **要提取的实体类型列表**: [IncidentType, ComplaintType, ProblemLevel, HandlingProcedure, Organization, Role, Penalty, Reward, Violation, ComplaintSource, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - CLASSIFIED_AS (分类为): 事件/投诉/违规归类为特定类型/级别。 (Violation/Event/Complaint -> IncidentType/ComplaintType/ProblemLevel)
          - HANDLED_BY_PROCEDURE (按流程处理): 事件/投诉/问题按哪个流程处理。 (IncidentType/ComplaintType/ProblemLevel/Violation -> HandlingProcedure)
          - LEADS_TO_PENALTY (导致处罚): 违规/问题级别导致的处罚。 (Violation/ProblemLevel -> Penalty)
          - LEADS_TO_REWARD (获得奖励): 特定行为/表现获得的奖励。 (Action/Performance -> Reward)
          - DETERMINES_ACCOUNTABILITY (确定责任): 处理流程确定了谁的责任。 (HandlingProcedure/Investigation -> Organization/Role)
          - APPLIES_PENALTY (执行处罚): 谁对谁/什么执行了处罚。 (Organization/Role -> Penalty -> Organization/Role/Violation)
          - RECEIVES_COMPLAINT_FROM (收到投诉): 处理流程/组织收到的投诉来源。 (HandlingProcedure/Organization -> ComplaintSource)
          # ... (Include relevant base relations like RESPONSIBLE_FOR handling) ...
        template: |
          请提取实体间的关系，**重点关注事件/投诉的分类 (CLASSIFIED_AS)，处理流程 (HANDLED_BY_PROCEDURE)，违规导致的处罚 (LEADS_TO_PENALTY)，责任认定 (DETERMINES_ACCOUNTABILITY) 以及处罚的执行 (APPLIES_PENALTY)**。

          {definitions}
          **要提取的关系类型列表**: [CLASSIFIED_AS, HANDLED_BY_PROCEDURE, LEADS_TO_PENALTY, LEADS_TO_REWARD, DETERMINES_ACCOUNTABILITY, APPLIES_PENALTY, RECEIVES_COMPLAINT_FROM, RESPONSIBLE_FOR, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Information_Reporting_Communication:
    schema_extension:
      entity_types:
        - ReportingRequirement # Re-emphasized
        - InformationType
        - ReportFormat
        - TimeLimit # Can be property or specific type
        - CommunicationChannel
        - ConfidentialityRequirement # Specific Requirement type
        - Recipient # Org/Role receiving info
      relation_types:
        - HAS_REPORTING_REQ # (InformationType/Event/Process -> ReportingRequirement)
        - SPECIFIES_FORMAT # (ReportingRequirement -> ReportFormat)
        - HAS_TIMELIMIT # (ReportingRequirement/ProcessStep -> TimeLimit)
        - USES_CHANNEL # (ReportingRequirement/Communication -> CommunicationChannel)
        - REQUIRES_CONFIDENTIALITY # (InformationType/ReportingRequirement -> ConfidentialityRequirement)
        - REPORTS_TO # (Organization/Role/Process -> Recipient/Organization/Role) - Re-emphasized
        - COMMUNICATES_WITH # (Organization/Role -> Organization/Role) - Re-emphasized
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - ReportingRequirement (报告要求): 关于信息报告的具体规定。
          - InformationType (信息类型): 如运营信息、安全信息。
          - ReportFormat (报告格式): 要求的报告形式或模板。
          - TimeLimit (时限): 报告或沟通的时间限制。
          - CommunicationChannel (沟通渠道): 如电话、系统、会议。
          - ConfidentialityRequirement (保密要求): 信息保密规定。
          - Recipient (接收方): 信息报告的目标单位或角色 (用Organization/Role)。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注各类信息的报告要求、报告路径、时限、格式、内容要求、保密要求以及沟通协调机制**。

          {definitions}
          **要提取的实体类型列表**: [ReportingRequirement, InformationType, ReportFormat, TimeLimit, CommunicationChannel, ConfidentialityRequirement, Organization, Role, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - HAS_REPORTING_REQ (有报告要求): 什么信息/事件/流程有报告要求。 (InformationType/Event/Process -> ReportingRequirement)
          - SPECIFIES_FORMAT (指定格式): 报告要求的格式。 (ReportingRequirement -> ReportFormat)
          - HAS_TIMELIMIT (有时限): 报告/步骤的时限。 (ReportingRequirement/ProcessStep -> TimeLimit)
          - USES_CHANNEL (使用渠道): 报告/沟通使用的渠道。 (ReportingRequirement/Communication -> CommunicationChannel)
          - REQUIRES_CONFIDENTIALITY (要求保密): 信息/报告的保密要求。 (InformationType/ReportingRequirement -> ConfidentialityRequirement)
          - REPORTS_TO (汇报给): 向谁汇报信息。 (Organization/Role/Process -> Organization/Role)
          - COMMUNICATES_WITH (沟通): 内部沟通协调关系。 (Organization/Role -> Organization/Role)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注报告的要求 (HAS_REPORTING_REQ)，包括格式 (SPECIFIES_FORMAT)、时限 (HAS_TIMELIMIT)、渠道 (USES_CHANNEL)、保密性 (REQUIRES_CONFIDENTIALITY)，以及报告路径 (REPORTS_TO) 和沟通机制 (COMMUNICATES_WITH)**。

          {definitions}
          **要提取的关系类型列表**: [HAS_REPORTING_REQ, SPECIFIES_FORMAT, HAS_TIMELIMIT, USES_CHANNEL, REQUIRES_CONFIDENTIALITY, REPORTS_TO, COMMUNICATES_WITH, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Legal_Regulatory_References:
    schema_extension:
      entity_types:
        - DocumentReference # Re-emphasized
        - LegalBasis # Specific Statement/Requirement type
        - ComplianceRequirement # Specific Requirement type
        - ExternalOrganization # Org outside the company, e.g., government body
      relation_types:
        - BASED_ON_DOC # (Document/Section/Rule -> DocumentReference) - More specific than REFERENCES
        - PROVIDES_LEGAL_BASIS # (DocumentReference -> LegalBasis/Rule)
        - MANDATES_COMPLIANCE # (DocumentReference/LegalBasis -> ComplianceRequirement)
        - ISSUED_BY # (DocumentReference -> ExternalOrganization)
        # REFERENCES (base) is also relevant
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - DocumentReference (外部文档引用): 引用的外部法律、法规、文件名称。
          - LegalBasis (法律依据): 作为制定依据的法律条款。
          - ComplianceRequirement (合规要求): 源自外部法规的具体合规要求。
          - ExternalOrganization (外部组织): 发布法规的政府部门或上级单位。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注引用的外部法律、法规、文件，以及作为依据的条款和相关的合规要求**。

          {definitions}
          **要提取的实体类型列表**: [DocumentReference, LegalBasis, ComplianceRequirement, ExternalOrganization, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - BASED_ON_DOC (依据文档): 规章/条款依据的外部文档。 (Document/Section/Rule -> DocumentReference)
          - PROVIDES_LEGAL_BASIS (提供法律依据): 外部文档提供了法律依据。 (DocumentReference -> LegalBasis/Rule)
          - MANDATES_COMPLIANCE (强制合规): 外部文档/依据强制的合规要求。 (DocumentReference/LegalBasis -> ComplianceRequirement)
          - ISSUED_BY (由...发布): 外部文档的发布机构。 (DocumentReference -> ExternalOrganization)
          - REFERENCES (引用): 通用引用关系。 (Document/Section -> DocumentReference)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注规章依据了哪些外部文件 (BASED_ON_DOC)，这些文件提供了什么依据 (PROVIDES_LEGAL_BASIS) 或合规要求 (MANDATES_COMPLIANCE)，以及文件由谁发布 (ISSUED_BY)**。

          {definitions}
          **要提取的关系类型列表**: [BASED_ON_DOC, PROVIDES_LEGAL_BASIS, MANDATES_COMPLIANCE, ISSUED_BY, REFERENCES, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

  Supplementary_Provisions:
    schema_extension:
      entity_types:
        - EffectiveDate
        - ExpiryDate
        - Interpreter # Use Organization/Role
        - AmendmentHistory # Can be Statement or specific type
        - AppendixReference # Specific DocumentReference type
      relation_types:
        - HAS_EFFECTIVE_DATE # (Document -> EffectiveDate)
        - HAS_EXPIRY_DATE # (Document -> ExpiryDate)
        - INTERPRETED_BY # (Document -> Interpreter/Organization/Role)
        - HAS_AMENDMENT # (Document -> AmendmentHistory/Statement)
        - REFERENCES_APPENDIX # (Document/Section -> AppendixReference)
        - SUPERSEDES # (Document -> DocumentReference) - Re-emphasized
    prompts:
      entity_extraction:
        definitions: |
          实体类型定义:
          - EffectiveDate (生效日期): 规章开始生效的日期。
          - ExpiryDate (失效日期): 规章失效的日期。
          - Interpreter (解释单位): 负责解释规章的单位 (用Organization/Role)。
          - AmendmentHistory (修订历史): 关于规章修订的说明。
          - AppendixReference (附件引用): 对附件的引用。
          - DocumentReference (文档引用): 废止的旧规。
          # ... (Include relevant base types) ...
        template: |
          请从以下文本中提取实体，**重点关注规章的生效日期、失效日期、解释权归属、修订历史、附件引用以及废止的旧规**。

          {definitions}
          **要提取的实体类型列表**: [EffectiveDate, ExpiryDate, Organization, Role, AmendmentHistory, AppendixReference, DocumentReference, ...]

          # ... (Rest of standard template: context, content, output format) ...
      relation_extraction:
        definitions: |
          关系类型定义:
          - HAS_EFFECTIVE_DATE (生效日期为): 文档的生效日期。 (Document -> EffectiveDate)
          - HAS_EXPIRY_DATE (失效日期为): 文档的失效日期。 (Document -> ExpiryDate)
          - INTERPRETED_BY (由...解释): 文档由谁解释。 (Document -> Organization/Role)
          - HAS_AMENDMENT (有修订): 文档的修订历史说明。 (Document -> AmendmentHistory/Statement)
          - REFERENCES_APPENDIX (引用附件): 引用了哪个附件。 (Document/Section -> AppendixReference)
          - SUPERSEDES (取代): 本文档取代了哪个旧文档。 (Document -> DocumentReference)
          # ... (Include relevant base relations) ...
        template: |
          请提取实体间的关系，**重点关注文档的生效/失效日期 (HAS_EFFECTIVE_DATE/HAS_EXPIRY_DATE)，解释单位 (INTERPRETED_BY)，修订历史 (HAS_AMENDMENT)，附件引用 (REFERENCES_APPENDIX) 以及取代关系 (SUPERSEDES)**。

          {definitions}
          **要提取的关系类型列表**: [HAS_EFFECTIVE_DATE, HAS_EXPIRY_DATE, INTERPRETED_BY, HAS_AMENDMENT, REFERENCES_APPENDIX, SUPERSEDES, ...]
          # ... (Rest of standard template: context, content, entities, output format) ...

# 图数据库配置 (保持不变)
database:
  enable_uniqueness_constraints: true
  enable_relation_metadata: true


**说明:**

1.  **`category_configs` Section:** This new top-level key holds the configurations for each of the 15 categories.
2.  **`schema_extension`:** Under each category, this lists *additional* entity and relation types relevant to that category, beyond the base schema. Some types might be re-emphasized if they are particularly crucial for that category. I've added some context types (like `Equipment`, `TicketType`, `AssetStatus`) where helpful.
3.  **`prompts`:** Each category has its own `entity_extraction` and `relation_extraction` prompts.
    *   **`definitions`:** These include definitions for the *new* or *most relevant* types for that category. They should ideally also include relevant base types for completeness when presented to the LLM.
    *   **`template`:** The core instruction within the template is modified to guide the LLM's focus according to the "抽取侧重" you provided (e.g., "重点关注...", "focus on..."). The placeholders `{definitions}`, `{entity_types}`, `{relation_types}`, etc., would need to be populated dynamically based on the chosen category's configuration (base + extension).
4.  **Base Schema/Prompts:** The original sections remain as defaults or a foundation. A real implementation would need logic to decide whether to merge the category-specific config with the base or completely replace parts of it.
5.  **Placeholders:** The templates use placeholders like `{entity_types}`. In a running system, these would be filled with the actual list of types relevant for the *current* category being processed (e.g., base types + types from the category's `schema_extension`).
6.  **Simplifications:** For brevity, I haven't exhaustively listed *all* potentially relevant base types in every category's prompt definition, but a real implementation should include them. Also, some "抽取侧重" items (like `Scope`, `Responsibility`, `Accountability`) are often better represented by relations or existing types (`APPLIES_TO`, `RESPONSIBLE_FOR`, `Organization`/`Role`) rather than distinct new entity types, which is reflected in the design.

This modified structure provides a clear way to manage different schemas and prompts tailored to the specific content of different regulation categories, directly addressing the need you identified.
```



--------------------


<a href="https://chatboxai.app" style="display: flex; align-items: center;">
<img src='https://chatboxai.app/icon.png' style='width: 40px; height: 40px; padding-right: 6px'>
<b style='font-size:30px'>Chatbox AI</b>
</a>
