# Supported Model Tasks
TASK_OCR = "ocr"
TASK_TABLE = "table"
TASK_FORMULA = "formula"
TASK_CHART = "chart"

TASKS = [TASK_OCR, TASK_TABLE, TASK_FORMULA, TASK_CHART]

# Default model prompt instructions
TASK_PROMPTS = {
    TASK_OCR: "OCR:",
    TASK_TABLE: "Table Recognition:",
    TASK_FORMULA: "Formula Recognition:",
    TASK_CHART: "Chart Recognition:",
}
