# NPU Lab — экспериментальный набор для работы с NPU на Android

Этот набор документов написан как «введение в реальный NPU» для разработчика, который раньше работал в основном с CUDA/CPU. Никаких маркетинговых обещаний — только то, как устроена железка и как с ней разговаривает код.

## Что внутри

| Раздел | О чём                                                                                                       |
|--------|-------------------------------------------------------------------------------------------------------------|
| [01](01-npu-basics.md) | Что такое NPU, чем отличается от GPU, какие классы NPU бывают                                  |
| [02](02-snapdragon-hexagon.md) | Hexagon NPU в Snapdragon 8 Elite Gen 5 (S26 Ultra) — числа, архитектура                |
| [03](03-checking-npu.md) | Как посмотреть, что за NPU в твоём устройстве (adb, /proc, dumpsys)                           |
| [04](04-qnn-sdk-setup.md) | Установка Qualcomm AI Engine Direct (QNN) SDK                                                |
| [05](05-using-ai-hub.md) | Получение готовых моделей с Qualcomm AI Hub                                                   |
| [06](06-coding-npu.md) | Жизненный цикл QNN runtime в коде                                                              |
| [07](07-benchmarking.md) | Как корректно сравнить NPU с RTX 4080S                                                       |
| [08](08-alternative-onnx.md) | Альтернатива: ONNX Runtime + QNN EP (более портативный путь, на будущее)               |
| [09](09-troubleshooting.md) | Что делать когда не работает                                                              |
| [10](10-custom-models.md) | Свои модели на NPU: AI Hub и полностью локальная компиляция                                 |
| [11](11-vision-models.md) | Vision-вкладка: классификация/глубина/сегментация/SR по фото и с камеры                     |

## С чего начать

1. Прочитай [01-npu-basics.md](01-npu-basics.md) и [02-snapdragon-hexagon.md](02-snapdragon-hexagon.md) — это база.
2. Подключи телефон по USB и пройди [03-checking-npu.md](03-checking-npu.md) — убедись, что у тебя действительно тот чип, что ты ожидаешь.
3. Поставь QNN SDK по [04-qnn-sdk-setup.md](04-qnn-sdk-setup.md).
4. Достань модели через [05-using-ai-hub.md](05-using-ai-hub.md).
5. Собери и запусти приложение — корневой [README.ru.md](../README.ru.md).

## Терминология

- **NPU** (Neural Processing Unit) — обобщённое название процессора, заточенного под умножение матриц низкой точности (INT8/FP16) при низком энергопотреблении.
- **Hexagon** — линейка DSP-процессоров от Qualcomm. В современных Snapdragon отделена под именем **HTP** (Hexagon Tensor Processor) — это и есть NPU.
- **QNN** = **Qualcomm AI Engine Direct** — низкоуровневый C API для запуска моделей на любом из бэкендов чипа (HTP / Adreno GPU / Kryo CPU).
- **QAI Hub** = **Qualcomm AI Hub** — облачный сервис от Qualcomm, где можно скомпилировать свою модель в QNN context binary под конкретное устройство. Также раздаёт ~150 предкомпилированных моделей.
- **Context binary** — бинарник `.bin`, который содержит граф модели, веса и метаинформацию в формате QNN. Готов к загрузке на устройство одним вызовом `QnnContext_createFromBinary`.
