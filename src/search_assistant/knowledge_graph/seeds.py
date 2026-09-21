from __future__ import annotations

from search_assistant.contracts import (
    DomainKnowledgeEntity,
    DomainKnowledgeGraph,
    DomainKnowledgeRelation,
)


TUTORIAL_GRAPHRAG_REF = "tutorial:AI-Agents-in-Depth-zh-CN#3.3.1-GraphRAG"
TUTORIAL_MEMORY_REF = "tutorial:AI-Agents-in-Depth-zh-CN#3.1-3.4-memory-and-knowledge"
PROJECT_BRIEFING_REF = "project:README#Daily-Briefing-Contract"


DEFAULT_DOMAIN_GRAPHS: tuple[DomainKnowledgeGraph, ...] = (
    DomainKnowledgeGraph(
        id="agent-engineering",
        name="AI Agent 工程",
        description=(
            "面向 Agent 系统研发的领域图谱，描述模型之外的 Harness、上下文、工具、验证、"
            "记忆和 GraphRAG 之间的工程关系。"
        ),
        overview=(
            "该图谱采用教程中的 GraphRAG 思路：把 Agent 工程知识拆成实体与关系，"
            "同时保留自然语言摘要，避免三元组丢失 Harness 设计原则、故障边界和检索策略。"
        ),
        source="reviewed-seed:agent-engineering",
        entities=[
            DomainKnowledgeEntity(
                id="harness-engineering",
                name="Harness 工程",
                entity_type="concept",
                aliases=["Agent Harness", "模型之外的工程层", "执行支架", "agent harness engineering", "harness engineering"],
                summary=(
                    "围绕模型构建的上下文、工具、权限、验证、纠错和可观测性工程层；"
                    "它决定 Agent 是否能稳定完成真实任务，而不只是生成一次性文本。"
                ),
                evidence_refs=[TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "semantic", "tutorial_pattern": "advanced-json-card"},
            ),
            DomainKnowledgeEntity(
                id="context-management",
                name="上下文管理",
                entity_type="workflow",
                aliases=["Context Engineering", "上下文工程", "context management", "context engineering"],
                summary=(
                    "负责把用户输入、系统状态、检索结果和工具观察组织成模型可使用的信息视图，"
                    "并通过摘要、索引或按需加载控制上下文成本。"
                ),
                evidence_refs=[TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="tool-loop",
                name="工具调用循环",
                entity_type="workflow",
                aliases=["ReAct 循环", "Action-Observation loop", "工具执行循环", "tool loop", "tool invocation loop", "action observation loop"],
                summary=(
                    "Agent 通过思考、调用工具、观察结果、再决定下一步的循环完成任务；"
                    "需要清晰工具边界、参数约束和失败处理。"
                ),
                evidence_refs=[TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="verification-correction",
                name="验证与纠正",
                entity_type="control",
                aliases=["Guardrails", "结果校验", "纠错机制", "verification and correction", "guardrails mechanism"],
                summary=(
                    "对工具返回、模型输出和最终结论进行类型检查、证据核验、重试、降级或人工接管，"
                    "用于闭合 Agent 的可靠性缺口。"
                ),
                evidence_refs=[TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="agentic-rag",
                name="智能体化 RAG",
                entity_type="workflow",
                aliases=["Agentic RAG", "迭代检索", "主动检索", "agentic rag", "iterative retrieval", "active retrieval"],
                summary=(
                    "让 Agent 自主判断搜索什么、信息是否足够、是否需要二次检索；"
                    "适合多跳、复杂、信息需求不明确的问题。"
                ),
                evidence_refs=[TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="graphrag",
                name="GraphRAG",
                entity_type="concept",
                aliases=["知识图谱检索", "实体关系图谱", "Graph-based RAG", "graph rag", "knowledge graph retrieval", "entity relation graph"],
                summary=(
                    "把文档或记忆提炼成实体-关系网络，用三元组表达可遍历关系；"
                    "擅长多跳推理和实体消歧，但应与自然语言摘要并存以保留完整语义。"
                ),
                evidence_refs=[TUTORIAL_GRAPHRAG_REF],
                metadata={"memory_type": "semantic", "tutorial_pattern": "entity-relation-index"},
            ),
            DomainKnowledgeEntity(
                id="memory-system",
                name="持久化记忆系统",
                entity_type="system",
                aliases=["长期记忆", "用户记忆", "领域知识库", "persistent memory", "agent memory", "long-term memory", "domain knowledge base"],
                summary=(
                    "将跨会话事实、领域知识、事件和流程以可审查结构保存，"
                    "再通过检索或图遍历激活到当前任务上下文。"
                ),
                evidence_refs=[TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "semantic"},
            ),
        ],
        relations=[
            DomainKnowledgeRelation(
                id="harness-contains-context",
                source_entity_id="harness-engineering",
                relation_type="contains",
                target_entity_id="context-management",
                description="Harness 工程通过上下文管理保证模型在每个决策点看到足够且不过载的信息。",
                evidence_refs=[TUTORIAL_MEMORY_REF],
            ),
            DomainKnowledgeRelation(
                id="harness-contains-tool-loop",
                source_entity_id="harness-engineering",
                relation_type="contains",
                target_entity_id="tool-loop",
                description="工具调用循环是 Harness 把模型推理连接到外部能力的执行层。",
                evidence_refs=[TUTORIAL_MEMORY_REF],
            ),
            DomainKnowledgeRelation(
                id="harness-closed-by-verification",
                source_entity_id="harness-engineering",
                relation_type="closed_by",
                target_entity_id="verification-correction",
                description="验证与纠正把 Agent 的执行偏差转化为重试、降级或人工接管。",
                evidence_refs=[TUTORIAL_MEMORY_REF],
            ),
            DomainKnowledgeRelation(
                id="agentic-rag-uses-tool-loop",
                source_entity_id="agentic-rag",
                relation_type="uses",
                target_entity_id="tool-loop",
                description="智能体化 RAG 使用工具循环多轮检索、观察并判断信息是否充分。",
                evidence_refs=[TUTORIAL_MEMORY_REF],
            ),
            DomainKnowledgeRelation(
                id="graphrag-indexes-memory",
                source_entity_id="graphrag",
                relation_type="indexes",
                target_entity_id="memory-system",
                description="GraphRAG 可作为长期记忆和共享知识库的结构化索引，用于多跳关系导航。",
                evidence_refs=[TUTORIAL_GRAPHRAG_REF],
            ),
            DomainKnowledgeRelation(
                id="context-loads-memory",
                source_entity_id="context-management",
                relation_type="loads",
                target_entity_id="memory-system",
                description="上下文管理按需激活长期记忆概览或细节，避免把全量知识塞进提示词。",
                evidence_refs=[TUTORIAL_MEMORY_REF],
            ),
        ],
    ),
    DomainKnowledgeGraph(
        id="industrial-ai",
        name="工业 AI",
        description="面向制造与工业现场的 AI 系统图谱，突出设备事件、MES、质检、追溯和人工接管。",
        overview=(
            "该图谱把工业 AI 先建模为受控工作流，而不是自主控制系统："
            "模型可以解释事件和生成建议，但写回生产系统必须经过接口、权限、验证和人工接管。"
        ),
        source="reviewed-seed:industrial-ai",
        entities=[
            DomainKnowledgeEntity(
                id="equipment-event",
                name="设备事件",
                entity_type="data",
                aliases=["机器事件", "传感器事件", "产线事件", "equipment event", "machine event", "sensor event"],
                summary="来自设备、传感器或产线系统的状态变化，是工业 AI 判断异常、生成建议或触发工单的输入。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="mes",
                name="MES",
                entity_type="system",
                aliases=["制造执行系统", "Manufacturing Execution System"],
                summary="承接生产计划、工单执行、物料和质量记录的制造执行系统，是工业 AI 结果写回的高风险边界。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="controlled-work-order-orchestration",
                name="受控工单编排",
                entity_type="workflow",
                aliases=["工单生成", "候选工单", "受控编排", "controlled work order orchestration", "work order generation"],
                summary=(
                    "把设备事件、历史规则和模型建议转成候选工单，再经权限、审批和回滚机制决定是否写入 MES。"
                ),
                evidence_refs=[PROJECT_BRIEFING_REF, TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="machine-vision-quality-inspection",
                name="机器视觉质检",
                entity_type="workflow",
                aliases=["视觉检测", "缺陷检测", "工业视觉", "machine vision quality inspection", "visual inspection", "defect detection"],
                summary="用图像、视频或多传感数据识别缺陷、尺寸偏差和工艺异常，是质量闭环的数据入口之一。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="quality-traceability",
                name="质量追溯",
                entity_type="system",
                aliases=["追溯链", "质量记录", "批次追溯", "quality traceability", "batch traceability", "quality record"],
                summary="把缺陷、工序、物料批次、设备状态和处置动作串联起来，用于定位质量问题来源和复盘处置效果。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="human-approval-rollback",
                name="人工审批与回滚",
                entity_type="control",
                aliases=["人工接管", "失败回滚", "审批链", "human approval rollback", "manual takeover", "rollback mechanism"],
                summary="在模型建议影响生产系统前提供授权、拒绝、回滚和审计路径，是工业 AI 进入真实流程的安全条件。",
                evidence_refs=[PROJECT_BRIEFING_REF, TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="production-data-interface",
                name="生产数据接口",
                entity_type="system",
                aliases=["系统接口", "数据写回", "API 适配器", "production data interface", "api adapter", "data writeback"],
                summary="连接设备、质检、MES、ERP 或数据湖的接口层，决定 AI 建议能否被安全、可追踪地执行。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
        ],
        relations=[
            DomainKnowledgeRelation(
                id="equipment-event-triggers-orchestration",
                source_entity_id="equipment-event",
                relation_type="triggers",
                target_entity_id="controlled-work-order-orchestration",
                description="设备事件触发异常解释、建议生成或候选工单编排。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="orchestration-writes-mes",
                source_entity_id="controlled-work-order-orchestration",
                relation_type="writes_to",
                target_entity_id="mes",
                description="候选工单只有通过权限、审批和验证后才能写回 MES。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="approval-guards-mes",
                source_entity_id="human-approval-rollback",
                relation_type="guards",
                target_entity_id="mes",
                description="人工审批与回滚机制守住 MES 写回边界，避免模型建议直接变成生产动作。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="vision-produces-traceability",
                source_entity_id="machine-vision-quality-inspection",
                relation_type="produces_evidence_for",
                target_entity_id="quality-traceability",
                description="视觉质检产生缺陷与工序证据，进入质量追溯链。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="interface-connects-orchestration",
                source_entity_id="production-data-interface",
                relation_type="connects",
                target_entity_id="controlled-work-order-orchestration",
                description="生产数据接口把设备事件和业务系统连接到工单编排流程。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
        ],
    ),
    DomainKnowledgeGraph(
        id="medical-imaging-ai",
        name="医学影像 AI",
        description="面向医学影像大模型和临床影像工作流的图谱，覆盖数据、系统接口、模型任务和临床验证。",
        overview=(
            "该图谱把医学影像 AI 建模为高风险、强证据约束的辅助诊断工作流："
            "模型输出必须关联影像标准、标注质量、临床验证、人机协同和隐私合规。"
        ),
        source="reviewed-seed:medical-imaging-ai",
        entities=[
            DomainKnowledgeEntity(
                id="multimodal-medical-imaging-model",
                name="多模态医学影像模型",
                entity_type="concept",
                aliases=["医学影像大模型", "影像基础模型", "vision-language medical model", "multimodal medical imaging model", "medical imaging foundation model"],
                summary="结合 CT、MRI、X 光、超声、报告文本或病历上下文进行表征学习和辅助分析的模型能力层。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="dicom-pacs",
                name="DICOM/PACS",
                entity_type="system",
                aliases=["DICOM", "PACS", "影像归档与通信系统", "dicom pacs", "picture archiving and communication system"],
                summary="医学影像数据与医院影像系统的核心标准和存取边界，决定模型接入和结果回写方式。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="annotation-quality-control",
                name="标注与质控",
                entity_type="control",
                aliases=["医生标注", "数据质控", "标签质量", "annotation quality control", "data quality control", "label quality"],
                summary="影像标注、病例筛选、标签一致性和质量控制直接影响模型训练、评估和临床可信度。",
                evidence_refs=[PROJECT_BRIEFING_REF, TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="lesion-detection-segmentation",
                name="病灶检测与分割",
                entity_type="workflow",
                aliases=["病灶识别", "影像分割", "检测任务", "lesion detection segmentation", "lesion identification", "image segmentation"],
                summary="将影像输入转化为病灶位置、轮廓、体积或风险提示，是医学影像 AI 最常见的任务形态之一。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="clinical-validation",
                name="临床验证",
                entity_type="metric",
                aliases=["临床试验", "外部验证", "真实世界验证", "clinical validation", "external validation", "real world validation"],
                summary="通过多中心、外部数据、医生对照和关键指标评估模型能否安全进入真实诊疗流程。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "procedural"},
            ),
            DomainKnowledgeEntity(
                id="privacy-compliance",
                name="隐私与合规",
                entity_type="risk",
                aliases=["数据脱敏", "合规审查", "医疗数据安全", "privacy compliance", "data desensitization", "medical data security"],
                summary="医学影像数据涉及患者隐私、院内权限和监管要求，限制数据流转、训练和部署方式。",
                evidence_refs=[PROJECT_BRIEFING_REF],
                metadata={"memory_type": "semantic"},
            ),
            DomainKnowledgeEntity(
                id="human-ai-reading-workflow",
                name="人机协同阅片",
                entity_type="workflow",
                aliases=["医生复核", "辅助诊断", "AI 阅片", "human ai reading workflow", "doctor review", "ai assisted diagnosis"],
                summary="AI 提供候选发现、测量或报告建议，最终由医生结合临床上下文复核并承担诊疗判断。",
                evidence_refs=[PROJECT_BRIEFING_REF, TUTORIAL_MEMORY_REF],
                metadata={"memory_type": "procedural"},
            ),
        ],
        relations=[
            DomainKnowledgeRelation(
                id="dicom-feeds-model",
                source_entity_id="dicom-pacs",
                relation_type="feeds",
                target_entity_id="multimodal-medical-imaging-model",
                description="DICOM/PACS 提供模型接入的影像数据和院内系统边界。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="annotation-supervises-model",
                source_entity_id="annotation-quality-control",
                relation_type="supervises",
                target_entity_id="multimodal-medical-imaging-model",
                description="高质量标注与病例筛选是模型训练和评估可信度的前提。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="model-supports-detection",
                source_entity_id="multimodal-medical-imaging-model",
                relation_type="supports",
                target_entity_id="lesion-detection-segmentation",
                description="医学影像模型输出病灶检测、分割、测量或风险提示。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="detection-enters-reading-workflow",
                source_entity_id="lesion-detection-segmentation",
                relation_type="enters",
                target_entity_id="human-ai-reading-workflow",
                description="检测与分割结果进入医生阅片流程，作为候选证据而不是自动诊断结论。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="validation-gates-workflow",
                source_entity_id="clinical-validation",
                relation_type="gates",
                target_entity_id="human-ai-reading-workflow",
                description="临床验证决定 AI 辅助阅片能否从实验环境进入真实临床工作流。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
            DomainKnowledgeRelation(
                id="privacy-constrains-data",
                source_entity_id="privacy-compliance",
                relation_type="constrains",
                target_entity_id="dicom-pacs",
                description="隐私与合规要求限制影像数据导出、训练、共享和结果回写。",
                evidence_refs=[PROJECT_BRIEFING_REF],
            ),
        ],
    ),
)


def default_domain_graphs(domain_ids: set[str] | None = None) -> list[DomainKnowledgeGraph]:
    graphs = list(DEFAULT_DOMAIN_GRAPHS)
    if domain_ids is None:
        return graphs
    return [graph for graph in graphs if graph.id in domain_ids]
