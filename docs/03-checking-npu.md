# 03. Как посмотреть характеристики NPU на устройстве

Прежде чем устанавливать SDK, имеет смысл подтвердить, что у тебя действительно тот чип, который ты ожидаешь. Особенно если телефон куплен в регионе, где Samsung продаёт Exynos-версии — там NPU другой.

## Включаем ADB

1. На телефоне: Settings → About phone → Software information → тапаем 7 раз по «Build number» → разработческий режим включён.
2. Settings → Developer options → USB debugging → ON.
3. Подключаем кабелем к компу. На телефоне всплывёт «Allow USB debugging?» — Always allow.

Проверяем:
```
adb devices
```

Если устройство не видно в WSL — нужно прокинуть USB-устройство в WSL через `usbipd`, или работать с adb-over-network (см. ниже). На голом Linux обычно достаточно `sudo usermod -aG plugdev $USER` + правильного udev-правила; для Samsung см. файл `51-android.rules` от Google.

## Базовые свойства SoC

```
adb shell getprop ro.soc.model            # ожидаем: SM8850 (8 Elite Gen 5) или SM8750 / SM8650
adb shell getprop ro.soc.manufacturer     # QTI
adb shell getprop ro.product.cpu.abi      # arm64-v8a
adb shell getprop ro.product.cpu.abilist  # arm64-v8a (на современных только)
adb shell getprop ro.board.platform       # pineapple / sun / ... — внутреннее имя платформы
adb shell getprop ro.hardware             # qcom
```

Что увидим на S26 Ultra:
```
[ro.soc.model]: [SM8850]
[ro.soc.manufacturer]: [QTI]
[ro.product.cpu.abi]: [arm64-v8a]
[ro.board.platform]: [sun2]
```

`SM8850` — это и есть Snapdragon 8 Elite Gen 5. `pineapple` был у 8 Gen 3, `sun` у 8 Elite Gen 4, `sun2` у Gen 5.

## Что про процессор сам говорит

```
adb shell cat /proc/cpuinfo | head -40
```

В выводе будут `processor : 0..7` блоки с `CPU implementer`, `CPU architecture`, и т.п. Это про CPU-кластеры (Oryon-V3 cores). NPU здесь не виден — потому что технически это отдельный DSP-сопроцессор.

Частоты CPU:
```
adb shell cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
adb shell cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_max_freq
```

На S26 Ultra prime core разгоняется до ~4.6 ГГц.

## Hexagon DSP

```
adb shell ls /vendor/dsp/cdsp/                    # драйвер DSP
adb shell ls /vendor/lib64/libQnn*.so             # библиотеки QNN, ставит вендор
adb shell ls /vendor/lib64/libcdsprpc.so          # CDSP RPC для общения с DSP
```

На современных Snapdragon-устройствах libQnnSystem.so / libQnnHtp.so уже лежат в `/vendor/lib64/`. Они принадлежат вендору телефона (Samsung), и обычно их можно использовать напрямую — но иногда версия там старее, чем нужно, и тогда мы кладём свежие копии в APK.

Список доступных версий HTP backend (по именам `.so` файлов):
```
adb shell 'ls /vendor/lib64/ | grep QnnHtp'
```

Должны увидеть что-то вроде:
```
libQnnHtp.so
libQnnHtpNetRunExtensions.so
libQnnHtpPrepare.so
libQnnHtpProfilingReader.so
libQnnHtpV81Stub.so          ← это stub для HTP v81, нужен нам
```

Если есть `V81Stub` — у тебя HTP v81 (как на S26 Ultra).
Если только `V75Stub` — это 8 Gen 3.
`V73Stub` — 8 Gen 2.

## DSP memory / VTCM

```
adb shell cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_governors
adb shell cat /sys/kernel/debug/clk/qcom-dsp-clk/clk_rate 2>/dev/null
```

(VTCM нельзя прочитать из юзерспейса напрямую без root — но размер можно посмотреть из ответа `QnnDevice_getPlatformInfo` через QNN.)

## Что говорит наша программа

После того как соберёшь и установишь NPU Lab APK, открой вкладку **Device**. Там увидишь:

- SoC model + manufacturer (из `Build.SOC_MODEL` / `Build.SOC_MANUFACTURER`)
- HTP arch — наш guesser (см. `device/DeviceInfo.kt`) определит по имени SoC
- Какие `libQnn*.so` фактически лежат в APK
- JSON от runtime после инициализации (build id, и т.п.)

## NPU работает? Самый быстрый smoke-test

После установки APK и проталкивания моделей на устройство (`adb push models/...`) запусти бенчмарк MobileNet-V3 — она крошечная, прогоняется за миллисекунды. Если на HTP backend получаешь `median=5–10 ms`, а на CPU `median=20–50 ms` — NPU работает. Если HTP backend выкидывает ошибку при инициализации — см. [09-troubleshooting.md](09-troubleshooting.md).

→ Дальше: [04-qnn-sdk-setup.md](04-qnn-sdk-setup.md) — ставим QNN SDK.
