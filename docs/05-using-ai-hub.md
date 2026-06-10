# 05. Где брать модели

Сначала — главное: **для популярных моделей AI Hub аккаунт не нужен**. Qualcomm выкладывает уже скомпилированные context binaries прямо на свой публичный S3 (рекламирует это как «HuggingFace mirror»), и они открыты для скачивания через curl/wget без регистрации.

Два пути:
1. **Public S3 (без аккаунта)** — для SD 1.5, Whisper Base, и десятков других моделей. Это default путь в нашем `scripts/fetch-models.py`.
2. **Qualcomm AI Hub (нужен аккаунт)** — для всего остального. Нужен, если хочется скомпилировать свою модель или взять то, чего нет в public mirror.

## Путь 1 — Public S3, no auth

Qualcomm хостит готовые `.bin` (QNN context binary) на `qaihub-public-assets.s3.us-west-2.amazonaws.com`. Прямые ссылки публичны, проверяются `curl -I`. Каталог: https://huggingface.co/qualcomm — справа в файлах каждой модели есть «Files and versions» с прямыми S3 URL'ами (HuggingFace тут используется как индексная страница, файлы лежат на S3).

### Что есть в public mirror для S26 Ultra

| Модель           | Точность  | Что включено в zip                             | Размер ~ |
|------------------|-----------|-------------------------------------------------|----------|
| SD 1.5           | w8a16     | text_encoder.bin + unet.bin + vae_decoder.bin   | 1.5 ГБ   |
| Whisper Base     | float     | encoder.bin + decoder.bin                       | 80 МБ    |
| Llama 3.2 3B     | w4a16     | 4 шарда context binary                          | 1.8 ГБ   |
| YOLOv11 detect   | w8a8      | model.bin                                       | 10 МБ    |
| MidasV2          | w8a8      | model.bin                                       | 60 МБ    |
| OpenAI CLIP      | w8a16     | encoder.bin                                     | 350 МБ   |
| ... | | | |

Точные URL'ы конструируются по шаблону:
```
https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/<model_id>/releases/<release>/<model_id>-qnn_context_binary-<precision>-qualcomm_snapdragon_8_elite_gen5.zip
```

Где `<model_id>` — например `stable_diffusion_v1_5`, `<precision>` — например `w8a16` или `float`.

### Скачиваем

```bash
cd /home/alex_melan/npu_experiments
source .venv/bin/activate  # если делал venv; иначе и без него работает — нет внешних deps
python scripts/fetch-models.py             # SD 1.5 + Whisper (default)
python scripts/fetch-models.py --models sd15
python scripts/fetch-models.py --models whisper
```

Что делает скрипт:
1. Стримит ZIP с S3.
2. Распаковывает нужные .bin внутри в `models/<имя>/<стадия>.bin`.
3. Для SD дополнительно подтягивает CLIP-токенизатор с HuggingFace (`openai/clip-vit-large-patch14`).

Никаких авторизаций. Ни AI Hub аккаунта, ни QPM, ни HuggingFace токена.

## Путь 2 — Qualcomm AI Hub (нужен аккаунт)

Нужно, когда:
- На public S3 нет нужной модели или нет варианта context_binary (только qnn_dlc — например, для Real-ESRGAN и MobileNet-V3 публикуют только DLC).
- Хочется скомпилировать СВОЮ PyTorch/ONNX модель под конкретное устройство.
- Нужны custom параметры квантизации, размер контекста или другая разметка input shapes.

### Регистрация

1. https://aihub.qualcomm.com → Sign in (Google / GitHub / email).
2. Profile → Settings → API token → Generate. Сохрани токен — больше его не покажут.
3. На машине:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install qai-hub qai-hub-models
   qai-hub configure --api_token <твой_токен>
   ```

### Скачивание через наш скрипт

```bash
python scripts/fetch-models.py --source aihub --models esrgan mobilenet
```

Что произойдёт:
1. Скрипт берёт PyTorch-обёртки из пакета `qai-hub-models` (он же содержит каталог).
2. Каждую обёртку отправляет в облако Qualcomm на compile_job с параметрами:
   - `target_runtime = qnn_context_binary` — выходной формат
   - `device = "Samsung Galaxy S26 Ultra"` — таргет (определяет HTP arch и калибровку)
   - `quantize_full_type = w8a16` — точность
3. Ждёт окончания компиляции (минуты в облаке).
4. Скачивает `.bin` и складывает в `models/`.

Для SD 1.5 — три job'а (text_encoder, unet, vae_decoder). Если решил пойти этим путём вместо public mirror'а — каждая занимает ~5–10 минут.

### Компиляция собственной модели

```python
import qai_hub as hub
import torch

class MyTinyModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0

model = MyTinyModel().eval()
sample = (torch.randn(1, 3, 224, 224),)

job = hub.submit_compile_job(
    model=torch.jit.trace(model, sample),
    input_specs={"x": (sample[0].shape, "float32")},
    device=hub.Device("Samsung Galaxy S26 Ultra"),
    options="--target_runtime qnn_context_binary --quantize_full_type w8a16",
)
print(job.url)
compiled = job.wait()
compiled.download(filename="my_model.bin")
```

`--quantize_full_type` варианты:
- `w8a8` — самая агрессивная квантизация (классификаторы / детекторы).
- `w8a16` — стандарт, хорошо для генеративных задач.
- (без флага) — FP16, если квантизация не нужна.

### Цена

Бесплатный тариф AI Hub: 25 compile jobs/день, 100 profile jobs/месяц. Достаточно.

## Пересылка на устройство

После того как `models/` собрана любым из путей:
```bash
adb push models /sdcard/Android/data/io.melan.npulab/files/
```

Приложение видит модели в `getExternalFilesDir(null)/models/`. После push открой вкладку Device — там должно появиться зелёное ✓ напротив каждой модели в списке Installed.

> SD 1.5 — это ~1.8 ГБ (text_encoder + unet + vae_decoder). На S26 Ultra с 12/16 ГБ RAM и сотней свободных гигов — не проблема, но учти.

## ExecuTorch как ещё одна альтернатива

PyTorch предлагает свой path: [ExecuTorch + Qualcomm backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html). Это очень близко к нашему пути, но через PyTorch-native pipeline (Edge IR → ExecuTorch → QNN context binary). Если у тебя уже есть боевая PyTorch-модель и не хочется огибать AI Hub — это другой разумный вариант. Сейчас мы его не используем, чтобы не плодить пути в одном экспериментальном проекте, но это та же железяка под капотом.

→ Дальше: [06-coding-npu.md](06-coding-npu.md) — что происходит в коде после того, как `.bin` оказался на устройстве.
