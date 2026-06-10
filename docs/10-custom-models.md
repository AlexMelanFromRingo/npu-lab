# 10. Свои модели на NPU

Приложение умеет запускать **любой** QNN context binary: всё, что лежит в
`models/custom/*.bin` на устройстве, автоматически появляется на вкладке
**Benchmark**. Этот раздел — как превратить свою модель в такой `.bin`.

## TL;DR

```bash
# 1. Экспортируй модель в ONNX со СТАТИЧЕСКИМИ шейпами
# 2. Скомпилируй под SM8850 (Galaxy S26 Ultra):
python scripts/compile-model.py my_model.onnx                  # облако AI Hub (fp16)
python scripts/compile-model.py my_model.onnx --quantize w8a16 # + квантизация
python scripts/compile-model.py my_model.onnx --source local   # полностью оффлайн

# 3. Проверь, что бинарник под наш чип (socModel=87, dspArch=81):
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility \
    --context_binary models/custom/my_model.bin --json_file /tmp/meta.json

# 4. На телефон:
adb push models/custom /sdcard/Android/data/io.melan.npulab/files/models/

# 5. Benchmark → чип «my_model» появился в списке → Run.
```

## Путь A: Qualcomm AI Hub (рекомендуется)

Облачная компиляция: всегда свежий компилятор, правильная квантизация,
профилирование на реальных устройствах в облаке Qualcomm.

1. Бесплатный аккаунт: https://aihub.qualcomm.com → Settings → API Token.
2. `pip install qai-hub && qai-hub configure --api_token <ТОКЕН>`
3. `python scripts/compile-model.py my_model.onnx [--quantize w8a16]`

Принимает ONNX и TorchScript. Готовые архитектуры (то, что в каталоге
huggingface.co/qualcomm) проще брать через `qai-hub-models` — они уже знают
правильные опции экспорта (см. [05-using-ai-hub.md](05-using-ai-hub.md)).

## Путь B: локально, без облака

`scripts/compile-model.py --source local` гоняет цепочку SDK на твоём ПК:

```
ONNX ── qairt-converter ──▶ DLC ── qnn-context-binary-generator ──▶ .bin
                                    (libQnnHtp.so x86 + HTP extensions:
                                     soc_model=87, dsp_arch=v81)
```

Нюансы:
- SDK-биндинги требуют **Python 3.10** — скрипт сам поднимет `.venv-qairt`
  через `uv venv --python 3.10` (системный Python не трогается).
- Получается **fp16**-граф. Локальная квантизация возможна, но требует
  калибровочного датасета (`--input_list` у конвертера) — не автоматизировано.
- Проверено end-to-end: тестовая сеть из Conv/Relu/Gemm компилируется и
  читается интроспекцией (`socModel=87, dspArch=81`).

## Ограничения HTP — читать перед экспортом

| Ограничение | Что делать |
|---|---|
| Только **статические шейпы** | фиксируй размеры при экспорте ONNX (`dynamic_axes` — убрать) |
| Не все операторы есть на HTP | смотри лог компиляции; экзотика (custom ops, complex, fft) не пройдёт |
| Большие веса | w8a16/w4a16 квантизация на AI Hub; fp16 для < ~1 ГБ |
| context binary привязан к SoC | под другой чип — перекомпилировать (`--device`/`--chipset`) |
| 1 граф = 1 .bin в нашем приложении | многокомпонентные пайплайны (как SD) — отдельные .bin на стадию |

## Как приложение видит кастомные модели

`ModelStore.customAssets()` сканирует `models/custom/*.bin` при открытии
Benchmark. Имя чипа = имя файла. Бенчмарк прогоняет первый граф бинарника с
синтетическими входами (warmup + 8 итераций, median/p95). Для полноценного
UI-инференса (как Generate/Speech) нужен пайплайн под конкретную задачу —
смотри `WhisperPipeline.kt` как образец работы с KV-кэшами и fp16.
