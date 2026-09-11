"""
知心后端 · 冷启动脚本（开发文档 FAQ5）
用 AI 生成的"假人设"强行灌入 ChromaDB，使系统一启动就拥有丰富的测试记忆，
便于演示"前天说的话今天 AI 还记得"以及"跨场景模式识别"（赛题方向③）。
仅用于开发/演示，使用合成数据，不涉及任何真实用户。
"""
import sys
from pathlib import Path

# 允许直接以脚本方式运行（复用 main 的配置与集合）
sys.path.insert(0, str(Path(__file__).resolve().parent))
import main  # noqa: E402

# 5 份虚拟来访者档案（合成数据），每条为 {text, signals}
# signals: topic(话题) / sentiment(情绪极性) / hour(表达时段，用于深夜模式演示)
PROFILES = {
    # 赵某：工作话题反复 3 次 + 深夜低落 2 次 -> 演示 recurring_topic + late_night_low
    "seed_zhao": [
        {"text": "赵某，33岁职场人，高压项目，有房贷要还。", "signals": {"topic": "work", "sentiment": "neutral", "hour": 20}},
        {"text": "赵某说老板常年通宵布置任务，想辞职但不敢。", "signals": {"topic": "work", "sentiment": "negative", "hour": 23}},
        {"text": "赵某深夜加班后倾诉压力大到掉头发。", "signals": {"topic": "work", "sentiment": "negative", "hour": 1}},
    ],
    # 张三：考研话题反复 + 持续负面 -> 演示 recurring_topic + sustained_negative
    "seed_zhangsan": [
        {"text": "张三，22岁男，考研失败，昨晚没睡好，养了一只狗叫旺财。", "signals": {"topic": "study", "sentiment": "negative", "hour": 22}},
        {"text": "张三提到压力大时喜欢一个人静静，不喜欢被说'你要坚强'。", "signals": {"topic": "study", "sentiment": "negative", "hour": 21}},
        {"text": "张三最近在找调剂院校，情绪起伏较大。", "signals": {"topic": "study", "sentiment": "negative", "hour": 19}},
    ],
    # 李四：深夜失眠低落 -> 演示 late_night_low
    "seed_lisi": [
        {"text": "李四，45岁中年，创业破产，有一个上初中的女儿。", "signals": {"topic": "finance", "sentiment": "negative", "hour": 18}},
        {"text": "李四说最怕接到催债电话，晚上常失眠。", "signals": {"topic": "finance", "sentiment": "negative", "hour": 23}},
        {"text": "李四凌晨睡不着，感到撑不住。", "signals": {"topic": "health", "sentiment": "negative", "hour": 2}},
    ],
    # 王奶奶：普通家庭/健康话题，无模式命中（对照组）
    "seed_wangnai": [
        {"text": "王奶奶，72岁空巢老人，老伴已故，养了一只猫。", "signals": {"topic": "family", "sentiment": "neutral", "hour": 15}},
        {"text": "王奶奶说儿女在外地，平时很少说话的人。", "signals": {"topic": "family", "sentiment": "neutral", "hour": 16}},
        {"text": "王奶奶近期腰腿疼，去医院排队很折腾。", "signals": {"topic": "health", "sentiment": "negative", "hour": 10}},
    ],
    # 陈某：健康话题反复 -> 演示 recurring_topic
    "seed_chen": [
        {"text": "陈某，长期患慢性病，独自居住，较少社交。", "signals": {"topic": "health", "sentiment": "negative", "hour": 14}},
        {"text": "陈某说身体多处疼痛，情绪常低落。", "signals": {"topic": "health", "sentiment": "negative", "hour": 15}},
        {"text": "陈某提到不想给远亲添麻烦。", "signals": {"topic": "social", "sentiment": "negative", "hour": 16}},
    ],
    # 演示账号 demo_lin（比赛演示用固定 uid，二维码链接 ?uid=demo_lin 进入）：
    # 虚构人物"小林"——备考研究生的往届生，具体化记忆点保证 Top3 召回命中
    "demo_lin": [
        {"text": "用户小林，24岁往届生，正在二战考研，目标院校是哈尔滨工业大学。", "signals": {"topic": "study", "sentiment": "neutral", "hour": 20}},
        {"text": "小林养了一只叫豆豆的橘猫，压力大时会撸猫放松。", "signals": {"topic": "life", "sentiment": "positive", "hour": 21}},
        {"text": "小林说每天晚上十一点后最容易焦虑，担心辜负家里人的期待。", "signals": {"topic": "study", "sentiment": "negative", "hour": 23}},
    ],
}


def seed() -> int:
    # 先清理旧种子，保证幂等（重复运行不产生重复记忆）
    for user_id in PROFILES:
        try:
            main.memory_collection.delete(where={"user_id": user_id})
        except Exception:
            pass
    total = 0
    for user_id, memories in PROFILES.items():
        for item in memories:
            main.store_memory_sync(user_id, item["text"], signals=item["signals"])
            total += 1
    return total


if __name__ == "__main__":
    total = seed()
    print(f"冷启动完成：已灌入 {len(PROFILES)} 个虚拟用户、共 {total} 条记忆（含结构化信号）。")
    # 顺手自检：模式识别是否按预期命中（演示前自证）
    for uid in ("seed_zhao", "seed_zhangsan", "seed_lisi", "seed_wangnai"):
        import asyncio
        pats = asyncio.run(main.detect_patterns(uid))
        print(f"  {uid}: {[p['type'] for p in pats] or '无模式'}")
