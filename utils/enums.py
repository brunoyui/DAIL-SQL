class REPR_TYPE:
    CODE_REPRESENTATION = "SQL"
    CODE_REPRESENTATION_PT = "SQLPT"
    TEXT_REPRESENTATION = "TEXT"
    OPENAI_DEMOSTRATION = "NUMBERSIGN"
    OPENAI_DEMOSTRATION_PT = "NUMBERSIGNPT"
    BASIC = "BASELINE"
    ALPACA_SFT = "INSTRUCTION"
    OPENAI_DEMOSTRATION_WFK = "NUMBERSIGNWFK"
    BASIC_WOFK = "BASELINEWOFK"
    TEXT_REPRESENTATION_WFK = "TEXTWFK"
    ALPACA_SFT_WFK = "INSTRUCTIONWFK"
    OPENAI_DEMOSTRATION_WORULE = "NUMBERSIGNWORULE"
    CODE_REPRESENTATION_WRULE = "SQLWRULE"
    ALPACA_SFT_WRULE = "INSTRUCTIONWRULE"
    TEXT_REPRESENTATION_WRULE = "TEXTWRULE"
    CODE_REPRESENTATION_COT = "SQLCOT"
    TEXT_REPRESENTATION_COT = "TEXTCOT"
    OPENAI_DEMOSTRATION_COT = "NUMBERSIGNCOT"
    ALPACA_SFT_COT = "INSTRUCTIONCOT"
    CBR = "CBR"


class EXAMPLE_TYPE:
    ONLY_SQL = "ONLYSQL"
    QA = "QA"
    COMPLETE = "COMPLETE"
    QAWRULE = "QAWRULE"
    OPENAI_DEMOSTRATION_QA = "NUMBERSIGNQA"
    BASIC_QA = "BASELINEQA"
    

class SELECTOR_TYPE:
    COS_SIMILAR = "COSSIMILAR"
    RANDOM = "RANDOM"
    EUC_DISTANCE = "EUCDISTANCE"
    EUC_DISTANCE_THRESHOLD = "EUCDISTANCETHRESHOLD"
    EUC_DISTANCE_SKELETON_SIMILARITY_THRESHOLD = "EUCDISSKLSIMTHR"
    EUC_DISTANCE_QUESTION_MASK = "EUCDISQUESTIONMASK"
    EUC_DISTANCE_PRE_SKELETON_SIMILARITY_THRESHOLD = "EUCDISPRESKLSIMTHR"
    EUC_DISTANCE_PRE_SKELETON_SIMILARITY_PLUS = "EUCDISPRESKLSIMPLUS"
    EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD = "EUCDISMASKPRESKLSIMTHR"
    EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD_SHIFT = "EUCDISMASKPRESKLSIMTHRSHIFT"
    # TODO: from the same database


class LLM:
    # openai LLMs
    TEXT_DAVINCI_003 = "text-davinci-003"
    CODE_DAVINCI_002 = "code-davinci-002"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_35_TURBO_0301 = "gpt-3.5-turbo-0301"
    GPT_4 = "gpt-4"

    # LLMs that use openai completion api
    TASK_COMPLETIONS = [
        TEXT_DAVINCI_003,
        CODE_DAVINCI_002
    ]

    # LLMs that use openai chat api
    TASK_CHAT = [
        GPT_35_TURBO,
        GPT_35_TURBO_0613,
        GPT_35_TURBO_16K,
        GPT_35_TURBO_0301,
        GPT_4
    ]

    # LLMs that can run in batch
    BATCH_FORWARD = [
        TEXT_DAVINCI_003,
        CODE_DAVINCI_002
    ]

    costs_per_thousand = {
        TEXT_DAVINCI_003: 0.0200,
        CODE_DAVINCI_002: 0.0200,
        GPT_35_TURBO: 0.0020,
        GPT_35_TURBO_0613: 0.0020,
        GPT_35_TURBO_16K: 0.003,
        GPT_35_TURBO_0301: 0.0020,
        GPT_4: 0.03
    }

    # local LLMs
    LLAMA_7B = "llama-7b"
    ALPACA_7B = "alpaca-7b"
    # TONG_YI_QIAN_WEN = "qwen-v1"

    GEMINI_1_0_PRO = "gemini-1.0-pro"
    GEMINI_1_0_PRO_001 = "gemini-1.0-pro-001"
    GEMINI_1_5_PRO = "gemini-1.5-pro-latest"
