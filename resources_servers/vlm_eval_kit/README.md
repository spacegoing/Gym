# Description

## Relevant links
1. Publicly reported scores source: https://rank.opencompass.org.cn/leaderboard-multimodal
   1. Benchmark mapping https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb
2. Model config: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/config.py#L210
3. Dataset build call: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/run.py#L294
4. Run inference call: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/inference.py#L133
5. Judge model configs: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/run.py#L363

## Accuracy reproduction using GPT-4o-mini-20240718
|Benchmark|Name for run.py|Judge|Reported score|Original repo repro|Gym repro|
|---|---|---|---|---|---|
|MMBench V1.1|MMBench_DEV_EN_V11|N/A?|76|TODO|TODO|
|MMStar|MMStar|N/A|54.8|TODO|TODO|
|MMMU|MMMU_DEV_VAL|N/A|60|TODO|TODO|
|MathVista|MathVista_MINI|gpt-4o-mini|52.5|TODO|TODO|
|HallusionBench Avg.|HallusionBench|N/A|46.1|TODO|TODO|
|AI2D|AI2D_TEST,AI2D_TEST_NO_MASK|N/A|77.8|TODO|TODO|
|OCRBench|OCRBench|N/A|785|TODO|TODO|
|MMVet|MMVet|66.9|gpt-4-turbo|TODO|TODO|


Data links: ?

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
