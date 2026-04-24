#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI文自动检测脚本 v2.0 —— 基于平台审核机制的全面检测
用法: python3 ai_text_checker.py <章节文件路径> [--fix] [--detail]
功能:
  - 基础检测: 标点限制、禁用词、句式、描写、对话、节奏（原v1规则）
  - 平台检测: 词汇多样性、句长波动、短语重复、标点节奏、对白比例、情绪/感官词密度
  - 综合评分: 模拟平台AI倾向分（0-100），输出风险评估

v2.0 新增:
  - 词汇多样性（TTR）检测
  - 句长波动（标准差+变异系数）检测
  - 短语重复（3-gram重复率）检测
  - 标点节奏（标点间隔变异系数）检测
  - 对白比例检测
  - 情绪词/感官词密度检测
  - 综合AI倾向评分（模拟平台检测）
"""

import re
import sys
import os
import math
from collections import Counter

# ============================================================
# 规则配置（与 writing-guide.md AI文规避规则保持同步）
# ============================================================

# 一、标点符号限制
EM_DASH_LIMIT = 3          # 破折号（——）每章上限
ELLIPSIS_LIMIT = 5         # 省略号（……）每章上限
COLON_LIST_LIMIT = 0       # 冒号列表（正文中的"XXX："后接解释/列举）
QUOTE_ABUSE_LIMIT = 3      # 双引号滥用（给普通词语加引号表示"特殊含义"）

# 二、禁用词汇
BANNED_TRANSITION_WORDS = [
    "然而", "此外", "再者", "综上所述", "总而言之",
    "值得注意的是", "需要指出的是", "不可否认", "事实上",
    "显然", "由此可见", "基于此", "不难发现", "毋庸置疑", "众所周知",
]

BANNED_CLICHE_PHRASES = [
    "不由得一愣", "眉头紧锁", "嘴角微微上扬", "心如鹿撞",
    "怦然心动", "热血沸腾", "手心出汗", "心跳好像打鼓一样",
    "喉咙发干", "眼眶微红",
]

RESTRICTED_ABSTRACT_WORDS = [
    "时代", "命运", "羁绊", "灵魂", "深渊", "力量", "智慧", "撕裂", "压迫", "窒息",
]
ABSTRACT_WORD_LIMIT = 1  # 每种最多出现1次

# 三、句式限制
BANNED_PATTERNS_THREE_PART = [
    r"首先.*?其次.*?最后",
    r"一方面.*?另一方面",
]

# 四、描写限制
ADJECTIVE_LIMIT = 2  # 每句形容词上限
VERB_MODIFIER_LIMIT = 3  # 动词修饰词密度上限（每段）

# 五、对话限制
DIALOGUE_SAY_MODIFIERS = [
    "冷冷地说", "缓缓地说", "淡淡地说", "低声说", "轻声说",
    "大声说", "怒吼道", "喝道", "厉声道", "柔声道",
    "冷笑道", "讥笑道", "嘲笑道", "微笑着说", "苦笑着说",
]

# 六、节奏限制
LYRICAL_ENDING_PATTERNS = [
    r"心中暗暗发誓",
    r"一定会改变",
    r"期待着.*?到来",
    r"新的.*?开始了",
    r"望着.*?天空",
    r"望着.*?远方",
]

# 七、情绪词库（AI高频使用的情绪描写词）
EMOTION_WORDS = [
    "震惊", "惊讶", "愤怒", "悲伤", "恐惧", "绝望", "兴奋", "激动",
    "感动", "温暖", "寒冷", "孤独", "寂寞", "焦虑", "紧张", "不安",
    "欣喜", "苦涩", "无奈", "犹豫", "坚定", "温柔", "冷漠", "骄傲",
    "羞愧", "尴尬", "后悔", "心痛", "心酸", "心寒", "心惊", "心虚",
    "沉重", "轻松", "愉快", "烦躁", "郁闷", "沮丧", "惶恐", "惊恐",
    "恼怒", "恼火", "欣喜若狂", "心如刀割", "百感交集", "五味杂陈",
    "忐忑不安", "惴惴不安", "心潮澎湃", "热血沸腾", "怒火中烧",
    "悲从中来", "喜出望外", "大失所望", "如释重负",
]

# 八、感官词库（五感描写词）
SENSORY_WORDS = [
    # 视觉
    "刺眼", "耀眼", "昏暗", "漆黑", "明亮", "模糊", "清晰", "闪烁",
    "耀眼", "暗淡", "绚丽", "斑斓", "灰蒙蒙", "金灿灿", "白茫茫",
    # 听觉
    "刺耳", "沉闷", "清脆", "嘈杂", "寂静", "轰鸣", "嗡嗡", "沙沙",
    "叮当", "呼啸", "呜咽", "低沉", "尖锐", "悠扬",
    # 触觉
    "灼热", "冰冷", "滚烫", "刺骨", "柔软", "坚硬", "粗糙", "光滑",
    "黏腻", "湿润", "干燥", "冰冷刺骨", "灼烧般",
    # 嗅觉
    "腥味", "焦味", "清香", "刺鼻", "芬芳", "霉味", "血腥味", "铁锈味",
    # 味觉
    "苦涩", "甘甜", "辛辣", "酸涩", "咸腥",
]

# 九、AI高频"安全表达"词库（AI喜欢用的万能描写词）
AI_SAFE_EXPRESSIONS = [
    "缓缓", "慢慢", "轻轻", "静静", "默默", "悄悄", "微微", "淡淡",
    "深深", "渐渐", "缓缓地", "慢慢地", "轻轻地", "静静地", "默默地",
    "似乎", "仿佛", "好像", "宛如", "犹如", "如同",
    "不禁", "忍不住", "不由得", "情不自禁",
    "一股", "一种", "一丝", "一抹", "一道", "一缕", "一阵",
]


# ============================================================
# 工具函数
# ============================================================

def read_file(filepath):
    """读取文件，返回正文部分（去掉章节小结）"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # 分离正文和章节小结
    parts = content.split("---\n")
    main_text = parts[0] if parts else content
    # 去掉标题行
    lines = main_text.strip().split("\n")
    body_lines = []
    for line in lines:
        if line.startswith("# ") or line.startswith("## "):
            continue
        body_lines.append(line)
    return "\n".join(body_lines), content


def count_pattern(text, pattern):
    """统计匹配次数"""
    return len(re.findall(pattern, text))


def find_pattern_with_context(text, pattern, context_chars=20):
    """找到所有匹配位置，返回 (行号, 匹配内容, 上下文) 列表"""
    results = []
    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        for m in re.finditer(pattern, line):
            start = max(0, m.start() - context_chars)
            end = min(len(line), m.end() + context_chars)
            context = line[start:end]
            results.append((i, m.group(), context))
    return results


def get_sentences(text):
    """将文本分割为句子列表，返回 [(句子长度, 句子内容), ...]"""
    lines = text.split("\n")
    sentences = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # 按句号、问号、感叹号分割
        sents = re.split(r"[。？！]", stripped)
        for s in sents:
            s = s.strip()
            if len(s) >= 3:  # 忽略太短的
                sentences.append((len(s), s))
    return sentences


def get_chinese_chars(text):
    """提取所有中文字符"""
    return re.findall(r'[\u4e00-\u9fff]', text)


def get_chinese_words(text):
    """简易中文分词：按2-4字切分（不依赖jieba）"""
    chars = get_chinese_chars(text)
    # 单字
    words = list(chars)
    # 双字词
    for i in range(len(chars) - 1):
        words.append(chars[i] + chars[i+1])
    # 三字词
    for i in range(len(chars) - 2):
        words.append(chars[i] + chars[i+1] + chars[i+2])
    return words


# ============================================================
# 平台级检测函数（v2.0新增）
# ============================================================

class PlatformDetector:
    """模拟网文平台AI检测的多维度分析器"""

    def __init__(self, body_text):
        self.text = body_text
        self.sentences = get_sentences(body_text)
        self.lines = body_text.split("\n")
        self.chinese_chars = get_chinese_chars(body_text)
        self.scores = {}  # 各维度得分
        self.details = {}  # 各维度详细信息

    def detect_all(self):
        """执行全部平台级检测"""
        results = []
        results.extend(self.check_vocabulary_diversity())
        results.extend(self.check_sentence_length_variance())
        results.extend(self.check_ngram_repetition())
        results.extend(self.check_punctuation_rhythm())
        results.extend(self.check_dialogue_ratio())
        results.extend(self.check_emotion_density())
        results.extend(self.check_sensory_density())
        results.extend(self.check_ai_safe_expressions())
        return results

    def calculate_ai_score(self):
        """计算综合AI倾向分（0-100），模拟平台检测"""
        # 各维度权重（参考平台检测权重排序）
        weights = {
            "ngram_rep": 0.20,      # 短语重复 - 权重最高
            "sentence_var": 0.18,   # 句长波动
            "vocab_div": 0.15,      # 词汇多样性
            "cliche": 0.12,         # 模板化表达
            "punct_rhythm": 0.10,   # 标点节奏
            "dialogue": 0.08,       # 对白比例
            "emotion": 0.07,        # 情绪词密度
            "sensory": 0.05,        # 感官词密度
            "ai_safe": 0.05,        # AI安全表达
        }

        total_score = 0
        for dim, weight in weights.items():
            if dim in self.scores:
                total_score += self.scores[dim] * weight

        return min(100, round(total_score, 1))

    def get_risk_level(self, score):
        """根据AI倾向分返回风险等级"""
        if score < 20:
            return "🟢 低风险", "文本特征接近人类写作"
        elif score < 40:
            return "🟡 中低风险", "部分维度轻微偏离，建议微调"
        elif score < 60:
            return "🟠 中风险", "多个维度偏离人类写作模式，需要修改"
        elif score < 80:
            return "🔴 高风险", "强烈AI特征，平台很可能标记"
        else:
            return "🚫 极高风险", "几乎确定会被平台判定为AI生成"

    # --- 维度1: 词汇多样性 ---

    def check_vocabulary_diversity(self):
        """检测词汇多样性（TTR - Type-Token Ratio）"""
        issues = []
        chars = self.chinese_chars
        total = len(chars)

        if total < 100:
            self.scores["vocab_div"] = 30  # 文本太短，给中间分
            return issues

        unique_chars = len(set(chars))
        ttr = unique_chars / total

        # TTR评分：越高越好（越接近人类写作）
        # 人类写作TTR通常在0.35-0.55之间
        # AI写作TTR通常在0.20-0.35之间
        if ttr >= 0.45:
            score = 5
        elif ttr >= 0.35:
            score = 15
        elif ttr >= 0.25:
            score = 40
        elif ttr >= 0.20:
            score = 65
        else:
            score = 85

        self.scores["vocab_div"] = score
        self.details["vocab_div"] = {
            "ttr": round(ttr, 4),
            "unique_chars": unique_chars,
            "total_chars": total,
        }

        if ttr < 0.30:
            issues.append((
                "🟠", "平台·词汇多样性", 0,
                f"TTR={ttr:.4f}（独特字{unique_chars}/总字{total}），词汇重复率高",
                "增加词汇变化，避免同一字词反复使用。人类写作TTR通常>0.35"
            ))
        elif ttr < 0.35:
            issues.append((
                "🟡", "平台·词汇多样性", 0,
                f"TTR={ttr:.4f}，略低于人类写作水平",
                "适当增加词汇丰富度"
            ))

        return issues

    # --- 维度2: 句长波动 ---

    def check_sentence_length_variance(self):
        """检测句长波动（标准差+变异系数）"""
        issues = []
        lengths = [s[0] for s in self.sentences]

        if len(lengths) < 5:
            self.scores["sentence_var"] = 30
            return issues

        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_len if mean_len > 0 else 0  # 变异系数

        # 人类写作句长标准差通常在3.5±2.5
        # AI写作句长标准差通常约1.2
        if std_dev >= 5.0:
            score = 5
        elif std_dev >= 3.5:
            score = 10
        elif std_dev >= 2.5:
            score = 25
        elif std_dev >= 1.5:
            score = 50
        else:
            score = 80

        self.scores["sentence_var"] = score
        self.details["sentence_var"] = {
            "mean": round(mean_len, 1),
            "std_dev": round(std_dev, 2),
            "cv": round(cv, 4),
            "sentence_count": len(lengths),
        }

        if std_dev < 2.0:
            issues.append((
                "🟠", "平台·句长波动", 0,
                f"句长标准差={std_dev:.2f}（均值{mean_len:.1f}字），句式过于均匀",
                "制造长短句交替。人类写作标准差通常>3.5，AI通常约1.2"
            ))
        elif std_dev < 3.0:
            issues.append((
                "🟡", "平台·句长波动", 0,
                f"句长标准差={std_dev:.2f}，节奏变化不足",
                "增加短句（<10字）和长句（>30字）的穿插"
            ))

        return issues

    # --- 维度3: 短语重复(3-gram) ---

    def check_ngram_repetition(self):
        """检测3-gram（三字组合）重复率"""
        issues = []
        chars = self.chinese_chars

        if len(chars) < 50:
            self.scores["ngram_rep"] = 30
            return issues

        # 生成3-gram
        trigrams = []
        for i in range(len(chars) - 2):
            trigrams.append(chars[i] + chars[i+1] + chars[i+2])

        total_trigrams = len(trigrams)
        unique_trigrams = len(set(trigrams))
        rep_ratio = 1 - (unique_trigrams / total_trigrams)  # 重复率

        # 找出重复次数最多的3-gram
        trigram_counts = Counter(trigrams)
        top_repeated = trigram_counts.most_common(5)

        # 评分：重复率越低越好
        if rep_ratio <= 0.30:
            score = 5
        elif rep_ratio <= 0.45:
            score = 15
        elif rep_ratio <= 0.55:
            score = 35
        elif rep_ratio <= 0.65:
            score = 55
        else:
            score = 80

        self.scores["ngram_rep"] = score
        self.details["ngram_rep"] = {
            "rep_ratio": round(rep_ratio, 4),
            "total": total_trigrams,
            "unique": unique_trigrams,
            "top_repeated": [(g, c) for g, c in top_repeated if c > 1],
        }

        if rep_ratio > 0.60:
            issues.append((
                "🟠", "平台·短语重复", 0,
                f"3-gram重复率={rep_ratio:.4f}（{unique_trigrams}/{total_trigrams}），短语重复严重",
                f"最重复的3-gram: {', '.join(f'「{g}」×{c}' for g, c in top_repeated[:3] if c > 1)}"
            ))
        elif rep_ratio > 0.50:
            issues.append((
                "🟡", "平台·短语重复", 0,
                f"3-gram重复率={rep_ratio:.4f}，略偏高",
                "替换重复的短语表达，用不同方式描述相似内容"
            ))

        return issues

    # --- 维度4: 标点节奏 ---

    def check_punctuation_rhythm(self):
        """检测标点节奏（标点间隔变异系数）"""
        issues = []

        # 统计每句话中的标点数量（作为"标点间隔"的近似）
        punctuation_marks = re.findall(r'[，。？！；：、…—""''（）\[\]【】]', self.text)

        if len(punctuation_marks) < 10:
            self.scores["punct_rhythm"] = 30
            return issues

        # 按段落统计标点数量
        para_punct_counts = []
        for line in self.lines:
            stripped = line.strip()
            if not stripped:
                continue
            punct_count = len(re.findall(r'[，。？！；：、…—""''（）\[\]【】]', stripped))
            if punct_count > 0:
                para_punct_counts.append(punct_count)

        if len(para_punct_counts) < 3:
            self.scores["punct_rhythm"] = 30
            return issues

        mean_punct = sum(para_punct_counts) / len(para_punct_counts)
        if mean_punct > 0:
            punct_variance = sum((c - mean_punct) ** 2 for c in para_punct_counts) / len(para_punct_counts)
            punct_std = math.sqrt(punct_variance)
            punct_cv = punct_std / mean_punct
        else:
            punct_cv = 0

        # 标点CV自然区间约0.45，AI通常更均匀（CV更低）
        if punct_cv >= 0.50:
            score = 5
        elif punct_cv >= 0.40:
            score = 15
        elif punct_cv >= 0.30:
            score = 35
        elif punct_cv >= 0.20:
            score = 55
        else:
            score = 75

        self.scores["punct_rhythm"] = score
        self.details["punct_rhythm"] = {
            "cv": round(punct_cv, 4),
            "mean": round(mean_punct, 2),
            "std": round(punct_std, 2) if mean_punct > 0 else 0,
        }

        if punct_cv < 0.25:
            issues.append((
                "🟠", "平台·标点节奏", 0,
                f"标点间隔变异系数={punct_cv:.4f}，标点分布过于均匀",
                "人类写作有断裂、跳句、突然停顿。尝试增加短段落、单句成段"
            ))
        elif punct_cv < 0.35:
            issues.append((
                "🟡", "平台·标点节奏", 0,
                f"标点间隔变异系数={punct_cv:.4f}，节奏变化不足",
                "尝试增加段落长度的变化，制造停顿感"
            ))

        return issues

    # --- 维度5: 对白比例 ---

    def check_dialogue_ratio(self):
        """检测对白比例"""
        issues = []

        # 统计对白行数
        dialogue_lines = 0
        total_lines = 0
        dialogue_chars = 0
        total_chars = len(self.chinese_chars)

        for line in self.lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            total_lines += 1
            # 检测对话行（以引号开头或包含引号）
            if stripped.startswith('"') or stripped.startswith('"') or stripped.startswith('「') or stripped.startswith('『'):
                dialogue_lines += 1
                # 统计对话中的中文字符
                dialogue_content = re.findall(r'[""「」]([^""「」]+)[""「」]', stripped)
                for dc in dialogue_content:
                    dialogue_chars += len(get_chinese_chars(dc))

        if total_lines == 0 or total_chars == 0:
            self.scores["dialogue"] = 30
            return issues

        line_ratio = dialogue_lines / total_lines
        char_ratio = dialogue_chars / total_chars

        # 自然区间：5%-65%
        if 0.10 <= line_ratio <= 0.55:
            score = 5
        elif 0.05 <= line_ratio <= 0.65:
            score = 20
        elif line_ratio < 0.05:
            score = 50
        else:
            score = 60

        self.scores["dialogue"] = score
        self.details["dialogue"] = {
            "line_ratio": round(line_ratio, 4),
            "char_ratio": round(char_ratio, 4),
            "dialogue_lines": dialogue_lines,
            "total_lines": total_lines,
        }

        if line_ratio < 0.05:
            issues.append((
                "🟡", "平台·对白比例", 0,
                f"对白占比={line_ratio:.1%}（{dialogue_lines}/{total_lines}行），几乎没有对话",
                "增加人物对话，网文对白比例通常在10%-55%"
            ))
        elif line_ratio > 0.65:
            issues.append((
                "🟡", "平台·对白比例", 0,
                f"对白占比={line_ratio:.1%}，对话过多",
                "增加叙述和描写，平衡对话与叙述"
            ))

        return issues

    # --- 维度6: 情绪词密度 ---

    def check_emotion_density(self):
        """检测情绪词密度"""
        issues = []
        total_chars = len(self.chinese_chars)

        if total_chars < 100:
            self.scores["emotion"] = 30
            return issues

        emotion_count = 0
        emotion_found = []
        for word in EMOTION_WORDS:
            count = count_pattern(self.text, re.escape(word))
            if count > 0:
                emotion_count += count
                emotion_found.append((word, count))

        density = emotion_count / total_chars * 1000  # 每千字情绪词数

        # 人类写作：每千字约2-8个情绪词
        # AI写作：每千字约8-15个情绪词（情绪堆叠）
        if density <= 5:
            score = 5
        elif density <= 8:
            score = 15
        elif density <= 12:
            score = 35
        elif density <= 18:
            score = 55
        else:
            score = 75

        self.scores["emotion"] = score
        self.details["emotion"] = {
            "density": round(density, 2),
            "count": emotion_count,
            "per_1000": round(density, 1),
        }

        if density > 12:
            top_emotions = sorted(emotion_found, key=lambda x: x[1], reverse=True)[:5]
            issues.append((
                "🟠", "平台·情绪词密度", 0,
                f"情绪词密度={density:.1f}‰（{emotion_count}个/千字），情绪堆叠",
                f"高频情绪词: {', '.join(f'「{w}」×{c}' for w, c in top_emotions)}"
            ))
        elif density > 8:
            issues.append((
                "🟡", "平台·情绪词密度", 0,
                f"情绪词密度={density:.1f}‰，略偏高",
                "用动作和细节替代直接的情绪描写"
            ))

        return issues

    # --- 维度7: 感官词密度 ---

    def check_sensory_density(self):
        """检测感官词密度"""
        issues = []
        total_chars = len(self.chinese_chars)

        if total_chars < 100:
            self.scores["sensory"] = 30
            return issues

        sensory_count = 0
        sensory_found = []
        for word in SENSORY_WORDS:
            count = count_pattern(self.text, re.escape(word))
            if count > 0:
                sensory_count += count
                sensory_found.append((word, count))

        density = sensory_count / total_chars * 1000  # 每千字感官词数

        # 感官词密度需要适中：过低像说明书，过高像刻意堆砌
        # 合理区间：每千字3-10个
        if 3 <= density <= 10:
            score = 5
        elif density < 3:
            score = 40
        elif density <= 15:
            score = 25
        else:
            score = 60

        self.scores["sensory"] = score
        self.details["sensory"] = {
            "density": round(density, 2),
            "count": sensory_count,
            "per_1000": round(density, 1),
        }

        if density > 15:
            top_sensory = sorted(sensory_found, key=lambda x: x[1], reverse=True)[:5]
            issues.append((
                "🟠", "平台·感官词密度", 0,
                f"感官词密度={density:.1f}‰（{sensory_count}个/千字），感官描写堆砌",
                f"高频感官词: {', '.join(f'「{w}」×{c}' for w, c in top_sensory)}"
            ))
        elif density < 3:
            issues.append((
                "🟡", "平台·感官词密度", 0,
                f"感官词密度={density:.1f}‰，感官描写不足",
                "增加视觉、听觉、触觉等五感描写"
            ))

        return issues

    # --- 维度8: AI安全表达 ---

    def check_ai_safe_expressions(self):
        """检测AI高频'安全表达'词使用频率"""
        issues = []
        total_chars = len(self.chinese_chars)

        if total_chars < 100:
            self.scores["ai_safe"] = 30
            return issues

        safe_count = 0
        safe_found = []
        for word in AI_SAFE_EXPRESSIONS:
            count = count_pattern(self.text, re.escape(word))
            if count > 0:
                safe_count += count
                safe_found.append((word, count))

        density = safe_count / total_chars * 1000  # 每千字

        # AI安全表达密度：越低越好
        # 人类写作：每千字约3-8个
        # AI写作：每千字约10-20个
        if density <= 5:
            score = 5
        elif density <= 8:
            score = 15
        elif density <= 12:
            score = 35
        elif density <= 18:
            score = 55
        else:
            score = 75

        self.scores["ai_safe"] = score
        self.details["ai_safe"] = {
            "density": round(density, 2),
            "count": safe_count,
            "per_1000": round(density, 1),
        }

        if density > 12:
            top_safe = sorted(safe_found, key=lambda x: x[1], reverse=True)[:5]
            issues.append((
                "🟠", "平台·AI安全表达", 0,
                f"AI安全表达密度={density:.1f}‰（{safe_count}个/千字），AI痕迹明显",
                f"高频词: {', '.join(f'「{w}」×{c}' for w, c in top_safe)}"
            ))
        elif density > 8:
            issues.append((
                "🟡", "平台·AI安全表达", 0,
                f"AI安全表达密度={density:.1f}‰，略偏高",
                "替换「缓缓」「微微」「淡淡」等AI高频副词，用具体动作替代"
            ))

        return issues


# ============================================================
# 基础检测函数（原v1规则）
# ============================================================

class AItextChecker:
    def __init__(self, filepath):
        self.filepath = filepath
        self.body_text, self.full_text = read_file(filepath)
        self.issues = []  # (严重程度, 类别, 行号, 描述, 建议)
        self.stats = {}

    def check_all(self):
        """执行全部基础检测"""
        self.check_em_dash()
        self.check_ellipsis()
        self.check_colon_list()
        self.check_quote_abuse()
        self.check_banned_transitions()
        self.check_banned_cliches()
        self.check_restricted_abstract()
        self.check_three_part_patterns()
        self.check_verb_modifiers()
        self.check_dialogue_say_modifiers()
        self.check_lyrical_endings()
        self.check_deep_breath()
        self.check_hoarse_voice()
        self.check_sentence_length_variation()
        self.check_adjective_density()
        return self.issues

    # --- 一、标点符号 ---

    def check_em_dash(self):
        """检测破折号（——）"""
        lines = self.body_text.split("\n")
        locations = []
        for i, line in enumerate(lines, 1):
            if "——" in line:
                locations.append((i, line.strip()[:60]))
        count = len(locations)
        self.stats["破折号（——）"] = count
        if count > EM_DASH_LIMIT:
            for line_no, ctx in locations:
                self.issues.append((
                    "🔴", "标点", line_no,
                    f"破折号「——」(共{count}处，上限{EM_DASH_LIMIT})",
                    "用句号断句、逗号连接、或动作描写替代"
                ))

    def check_ellipsis(self):
        """检测省略号（……）"""
        count = count_pattern(self.body_text, "……")
        self.stats["省略号（……）"] = count
        if count > ELLIPSIS_LIMIT:
            matches = find_pattern_with_context(self.body_text, "……")
            for line_no, match, ctx in matches:
                self.issues.append((
                    "🔴", "标点", line_no,
                    f"省略号「{match}」(共{count}处，上限{ELLIPSIS_LIMIT})",
                    "用句号断句、动作描写、或「他顿了顿」替代"
                ))

    def check_colon_list(self):
        """检测冒号列表"""
        lines = self.body_text.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('"') or stripped.startswith('"'):
                continue
            if any(kw in line for kw in ["一种", "两种", "一种是", "另一种"]):
                continue
            if re.search(r"的[：:]", line):
                continue
            if re.search(r"[：:]\s*[\u4e00-\u9fff]", line):
                self.issues.append((
                    "🟡", "标点", i,
                    f"疑似冒号列表: 「{stripped[:50]}」",
                    "用句号断句，逗号连接替代"
                ))

    def check_quote_abuse(self):
        """检测双引号滥用（非对话用途的引号）"""
        lines = self.body_text.split("\n")
        count = 0
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('"') or stripped.startswith('"'):
                continue
            if "《" in line and "》" in line:
                continue
            if any(kw in line for kw in ["刻着", "写着", "落款", "匾额", "碑文", "字条"]):
                continue
            inner_quotes = re.findall(r'"([^"]{1,6})"', line)
            for q in inner_quotes:
                if re.match(r'^[\u4e00-\u9fff]{1,4}$', q):
                    count += 1
                    self.issues.append((
                        "🟡", "标点", i,
                        f"疑似双引号滥用: 「{q}」",
                        "如果是专有名词引用可保留，否则去掉引号"
                    ))
                else:
                    count += 1
                    self.issues.append((
                        "🟡", "标点", i,
                        f"双引号滥用: 「{q}」",
                        "去掉引号，直接写内容"
                    ))

    # --- 二、禁用词汇 ---

    def check_banned_transitions(self):
        """检测禁用的过渡/连接词"""
        for word in BANNED_TRANSITION_WORDS:
            matches = find_pattern_with_context(self.body_text, re.escape(word))
            for line_no, match, ctx in matches:
                self.issues.append((
                    "🔴", "禁用词", line_no,
                    f"禁用过渡词「{match}」",
                    f"删除或用口语化表达替代"
                ))

    def check_banned_cliches(self):
        """检测AI惯用描写套话"""
        for phrase in BANNED_CLICHE_PHRASES:
            matches = find_pattern_with_context(self.body_text, re.escape(phrase))
            for line_no, match, ctx in matches:
                self.issues.append((
                    "🔴", "禁用词", line_no,
                    f"AI套话「{match}」",
                    "用具体动作描写替代"
                ))

    def check_restricted_abstract(self):
        """检测受限的宏大/抽象词汇"""
        for word in RESTRICTED_ABSTRACT_WORDS:
            count = count_pattern(self.body_text, re.escape(word))
            if count > ABSTRACT_WORD_LIMIT:
                matches = find_pattern_with_context(self.body_text, re.escape(word))
                for line_no, match, ctx in matches:
                    self.issues.append((
                        "🟡", "抽象词", line_no,
                        f"抽象词「{match}」出现{count}次(上限{ABSTRACT_WORD_LIMIT})",
                        "用具体动作、细节、感官描写替代"
                    ))

    # --- 三、句式限制 ---

    def check_three_part_patterns(self):
        """检测三段式/排比三连"""
        for pattern in BANNED_PATTERNS_THREE_PART:
            matches = find_pattern_with_context(self.body_text, pattern, context_chars=40)
            for line_no, match, ctx in matches:
                self.issues.append((
                    "🔴", "句式", line_no,
                    f"三段式句式「{match[:30]}...」",
                    "拆分成自然的叙述，不要用排比结构"
                ))

    # --- 四、描写限制 ---

    def check_verb_modifiers(self):
        """检测动词修饰词密度"""
        lines = self.body_text.split("\n")
        for i, line in enumerate(lines, 1):
            modifiers = re.findall(r"[缓缓地|慢慢地|轻轻地|冷冷地|淡淡地|默默地|静静地|悄悄地|狠狠地|紧紧地|死死地]+[说看走想笑站坐握抬低转头]", line)
            if len(modifiers) >= 2:
                self.issues.append((
                    "🟡", "描写", i,
                    f"动词修饰词堆叠({len(modifiers)}个): 「{line.strip()[:50]}」",
                    "用动作替代状语，如「他放下筷子，说」替代「他缓缓地说」"
                ))

    def check_adjective_density(self):
        """检测形容词密度（每句≤2个，用"的"字密度近似）"""
        lines = self.body_text.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('"') or stripped.startswith('"'):
                continue
            adj_count = len(re.findall(r"的[\u4e00-\u9fff]", stripped))
            if adj_count >= 4:
                self.issues.append((
                    "🟡", "描写", i,
                    f"形容词密度较高({adj_count}个): 「{stripped[:50]}」",
                    "精简形容词，每句不超过2个"
                ))

    # --- 五、对话限制 ---

    def check_dialogue_say_modifiers(self):
        """检测对话中"说"的修饰词堆砌"""
        for modifier in DIALOGUE_SAY_MODIFIERS:
            matches = find_pattern_with_context(self.body_text, re.escape(modifier))
            for line_no, match, ctx in matches:
                self.issues.append((
                    "🟡", "对话", line_no,
                    f"「说」字修饰堆砌「{match}」",
                    "用动作替代，如「他没抬头」替代「他冷冷地说」"
                ))

    # --- 六、节奏限制 ---

    def check_lyrical_endings(self):
        """检测抒情式结尾"""
        lines = self.body_text.strip().split("\n")
        last_lines = lines[-5:] if len(lines) >= 5 else lines
        for i_offset, line in enumerate(last_lines):
            line_no = len(lines) - len(last_lines) + i_offset + 1
            for pattern in LYRICAL_ENDING_PATTERNS:
                if re.search(pattern, line):
                    self.issues.append((
                        "🔴", "节奏", line_no,
                        f"抒情式结尾「{line.strip()[:50]}」",
                        "用人物动作或对话结尾，如「朱标关上窗户，走回书桌前。」"
                    ))

    def check_deep_breath(self):
        """检测「深吸一口气」使用次数（每章最多1次）"""
        count = count_pattern(self.body_text, "深吸一口气")
        if count > 1:
            matches = find_pattern_with_context(self.body_text, "深吸一口气")
            for line_no, match, ctx in matches[1:]:
                self.issues.append((
                    "🟡", "禁用词", line_no,
                    f"「深吸一口气」出现{count}次(上限1)",
                    "保留1次，其余用其他动作替代"
                ))

    def check_hoarse_voice(self):
        """检测「声音沙哑」使用次数（每章最多1次）"""
        count = count_pattern(self.body_text, "声音沙哑")
        if count > 1:
            matches = find_pattern_with_context(self.body_text, "声音沙哑")
            for line_no, match, ctx in matches[1:]:
                self.issues.append((
                    "🟡", "禁用词", line_no,
                    f"「声音沙哑」出现{count}次(上限1)",
                    "保留1次，其余用其他方式描写"
                ))

    # --- 句长变化 ---

    def check_sentence_length_variation(self):
        """检测句长变化（相邻句子长度差应≥2:1）"""
        lines = self.body_text.split("\n")
        sentences = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            sents = re.split(r"[。？！]", stripped)
            for s in sents:
                s = s.strip()
                if len(s) >= 5:
                    sentences.append((len(s), s))

        similar_streak = 0
        for i in range(1, len(sentences)):
            prev_len = sentences[i-1][0]
            curr_len = sentences[i][0]
            ratio = max(prev_len, curr_len) / min(prev_len, curr_len) if min(prev_len, curr_len) > 0 else 1
            if ratio < 1.5:
                similar_streak += 1
                if similar_streak >= 3:
                    self.issues.append((
                        "🟡", "句式", 0,
                        f"连续{similar_streak + 1}个句子长度过于相近（缺乏节奏变化）",
                        "长短句交替，制造节奏感"
                    ))
                    similar_streak = 0
            else:
                similar_streak = 0


# ============================================================
# 报告输出
# ============================================================

def print_report(checker, platform_detector=None, detail_mode=False):
    """输出检测报告"""
    issues = checker.issues
    stats = checker.stats

    print("=" * 60)
    print(f"  AI文检测报告 v2.0: {os.path.basename(checker.filepath)}")
    print("=" * 60)

    # 标点统计
    print(f"\n📊 标点统计:")
    for name, count in stats.items():
        limit = EM_DASH_LIMIT if "破折号" in name else ELLIPSIS_LIMIT if "省略号" in name else "-"
        status = "✅" if (isinstance(limit, int) and count <= limit) else "❌"
        if isinstance(limit, int):
            print(f"   {status} {name}: {count}/{limit}")
        else:
            print(f"   {name}: {count}")

    # 问题分类统计
    critical = [i for i in issues if i[0] == "🔴"]
    warning = [i for i in issues if i[0] == "🟡"]
    platform_issues = [i for i in issues if i[0] in ("🟠", "🟡") and "平台" in i[1]]

    print(f"\n📋 基础检测问题汇总:")
    print(f"   🔴 严重问题: {len(critical)} 个")
    print(f"   🟡 建议修改: {len(warning)} 个")

    if not issues and not platform_detector:
        print(f"\n   ✅ 未检测到AI文特征，通过！")
        return

    # 按类别分组输出基础检测问题
    if issues:
        print(f"\n{'─' * 60}")
        categories = {}
        for issue in issues:
            cat = issue[1]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(issue)

        for cat, cat_issues in categories.items():
            print(f"\n【{cat}】({len(cat_issues)}个问题)")
            for severity, _, line_no, desc, suggestion in cat_issues:
                print(f"  {severity} 第{line_no}行 | {desc}")
                print(f"     💡 {suggestion}")

    # 平台级检测报告
    if platform_detector:
        print(f"\n{'═' * 60}")
        print(f"  🏢 平台级AI检测模拟")
        print(f"{'═' * 60}")

        # 各维度得分
        dim_names = {
            "vocab_div": "词汇多样性",
            "sentence_var": "句长波动",
            "ngram_rep": "短语重复(3-gram)",
            "punct_rhythm": "标点节奏",
            "dialogue": "对白比例",
            "emotion": "情绪词密度",
            "sensory": "感官词密度",
            "ai_safe": "AI安全表达",
        }

        print(f"\n📊 各维度评分（0=人类，100=AI）:")
        for dim_key, dim_name in dim_names.items():
            if dim_key in platform_detector.scores:
                score = platform_detector.scores[dim_key]
                if score < 20:
                    bar = "🟢" + "█" * 2 + "░" * 8
                elif score < 40:
                    bar = "🟢" + "█" * 4 + "░" * 6
                elif score < 60:
                    bar = "🟡" + "█" * 6 + "░" * 4
                elif score < 80:
                    bar = "🔴" + "█" * 8 + "░" * 2
                else:
                    bar = "🚫" + "█" * 10
                print(f"   {bar} {dim_name}: {score:5.1f}")

                # 详细信息
                if detail_mode and dim_key in platform_detector.details:
                    detail = platform_detector.details[dim_key]
                    detail_str = " | ".join(f"{k}={v}" for k, v in detail.items())
                    print(f"       └─ {detail_str}")

        # 综合评分
        ai_score = platform_detector.calculate_ai_score()
        risk_level, risk_desc = platform_detector.get_risk_level(ai_score)

        print(f"\n{'─' * 60}")
        print(f"  🎯 综合AI倾向分: {ai_score}/100")
        print(f"  {risk_level} {risk_desc}")
        print(f"{'─' * 60}")

        # 平台级问题
        platform_all_issues = platform_detector.detect_all()
        if platform_all_issues:
            print(f"\n📋 平台级问题({len(platform_all_issues)}个):")
            for severity, _, line_no, desc, suggestion in platform_all_issues:
                print(f"  {severity} | {desc}")
                if detail_mode:
                    print(f"     💡 {suggestion}")

    print(f"\n{'─' * 60}")
    total = len(issues) + (len(platform_detector.detect_all()) if platform_detector else 0)
    print(f"总计: {total} 个问题需要处理")


# ============================================================
# 自动修复（仅处理标点符号）
# ============================================================

def auto_fix_em_dash(text):
    """自动修复破折号：保留前3个，其余替换为句号或逗号"""
    count = 0
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        if count >= EM_DASH_LIMIT:
            if re.search(r'——$', line):
                line = re.sub(r'——$', '。', line)
            elif re.search(r'——"', line):
                line = re.sub(r'——"', '，"', line)
            else:
                line = line.replace("——", "，")
        else:
            count += line.count("——")
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def auto_fix_ellipsis(text):
    """自动修复省略号：保留前5个，其余替换"""
    count = 0
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        while "……" in line and count >= ELLIPSIS_LIMIT:
            if re.search(r'"……', line) or re.search(r'"……', line):
                line = re.sub(r'"……', '"，', line, count=1)
            else:
                line = line.replace("……", "。", 1)
        count += line.count("……")
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def auto_fix(filepath):
    """自动修复标点符号问题"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    content = auto_fix_em_dash(content)
    content = auto_fix_ellipsis(content)

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 已自动修复标点符号问题，保存到: {filepath}")
        return True
    else:
        print("ℹ️  标点符号已符合要求，无需修复")
        return False


# ============================================================
# 主入口
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python3 ai_text_checker.py <章节文件路径> [--fix] [--detail]")
        print("      python3 ai_text_checker.py <章节文件路径> --fix    (自动修复标点)")
        print("      python3 ai_text_checker.py <章节文件路径> --detail (显示详细数据)")
        print()
        print("示例: python3 ai_text_checker.py chapters/第001章-这不是梦.md")
        print("      python3 ai_text_checker.py chapters/第001章-这不是梦.md --fix")
        print("      python3 ai_text_checker.py chapters/第001章-这不是梦.md --detail")
        sys.exit(1)

    filepath = sys.argv[1]
    do_fix = "--fix" in sys.argv
    detail_mode = "--detail" in sys.argv

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)

    # 执行基础检测
    checker = AItextChecker(filepath)
    issues = checker.check_all()

    # 执行平台级检测
    platform_detector = PlatformDetector(checker.body_text)
    platform_issues = platform_detector.detect_all()

    # 合并所有问题
    all_issues = issues + platform_issues
    checker.issues = all_issues

    # 输出报告
    print_report(checker, platform_detector, detail_mode)

    # 自动修复
    if do_fix:
        print(f"\n{'=' * 60}")
        print("🔧 自动修复模式")
        print(f"{'=' * 60}")
        auto_fix(filepath)

        # 修复后重新检测
        print(f"\n{'=' * 60}")
        print("🔄 修复后重新检测")
        print(f"{'=' * 60}")
        checker2 = AItextChecker(filepath)
        pd2 = PlatformDetector(checker2.body_text)
        all_issues2 = checker2.check_all() + pd2.detect_all()
        checker2.issues = all_issues2
        print_report(checker2, pd2, detail_mode)


if __name__ == "__main__":
    main()
