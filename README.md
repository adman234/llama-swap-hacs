# llama-swap for Home Assistant

[![hacs][hacs-badge]][hacs-url]
[![validate][validate-badge]][validate-url]

A Home Assistant integration for [llama-swap][llama-swap], the model swapping
proxy for llama.cpp and friends. It exposes which model is loaded, the details
of every configured model, and the server's CPU/GPU load as entities you can
use in automations, and lets you load, unload and switch profiles.

Everything is local polling against llama-swap's own HTTP API. No cloud, no
extra dependencies.

## Installation

### HACS

1. In HACS, open the three-dot menu and choose **Custom repositories**.
2. Add `https://github.com/adman234/llama-swap-hacs` with the category
   **Integration**.
3. Search for **llama-swap**, install it, and restart Home Assistant.
4. Go to **Settings → Devices & Services → Add Integration** and pick
   **llama-swap**.

### Manual

Copy `custom_components/llama_swap` into your Home Assistant `config/custom_components`
directory and restart.

## Configuration

The config flow asks for:

| Field | Notes |
| --- | --- |
| Host | Hostname or IP of the machine running llama-swap |
| Port | llama-swap's listen port, `8080` by default |
| API key | Only if llama-swap is configured with `apiKey` |
| Use HTTPS | Tick if llama-swap sits behind TLS |
| Verify SSL certificate | Untick for a self-signed certificate |

The polling interval defaults to 15 seconds and can be changed under the
integration's **Configure** button.

## Entities

### The server

One device represents the llama-swap instance.

| Entity | What it gives you |
| --- | --- |
| `sensor.<server>_active_model` | The loaded model's ID, or unknown when nothing is loaded |
| `sensor.<server>_loaded_models` | How many models are loaded right now |
| `sensor.<server>_configured_models` | How many models llama-swap knows about |
| `binary_sensor.<server>_model_loaded` | On while any model holds a process |
| `select.<server>_profile` | The active profile, if profiles are configured |
| `button.<server>_unload_all_models` | Frees all loaded models |
| `sensor.<server>_version` | llama-swap build version (disabled by default) |
| `sensor.<server>_active_profile` | Active profile, as a plain sensor |

If llama-swap has performance monitoring enabled, you also get
`cpu_utilization`, `memory_used`, `memory_utilization`, `memory_total`,
`swap_used` and `load_average_1m` / `5m` / `15m`.

`sensor.<server>_active_model` reports one model ID. llama-swap can hold
several models at once when its groups allow it, so the full picture lives in
the attributes:

```yaml
loaded_models: [qwen3-coder, nomic-embed]
loaded_count: 2
models:
  - model_id: qwen3-coder
    type: model
    loaded: true
    ...
```

### Each model

Every model, alias target, selector and peer model llama-swap lists gets its
own device, linked to the server.

| Entity | What it gives you |
| --- | --- |
| `sensor.<model>_state` | `stopped`, `starting`, `ready`, `stopping` or `shutdown` |
| `binary_sensor.<model>_loaded` | On while the model holds a process |
| `switch.<model>_loaded` | Turn on to load, off to unload |
| `button.<model>_unload` | Unload just this model |
| `sensor.<model>_context_length` | Configured context size, when declared |
| `sensor.<model>_unload_after` | The model's TTL in seconds (disabled by default) |

The state sensor carries the model's full detail as attributes, which is what
you want for templates and conditions:

`model_id`, `model_name`, `description`, `type` (`model`, `alias`, `selector`,
`peer` or `profile`), `aliases`, `capabilities` (for example
`{"vision": true, "function_calling": true}`), `architecture` (input and output
modalities), `supported_parameters`, `context_length`, `cmd`, `proxy`, `ttl`,
`unlisted`, `loaded`, and any custom `metadata` from your llama-swap config.

> The `cmd` attribute is llama-swap's upstream command line. It shows model
> paths and launch flags. It is redacted from downloadable diagnostics, but it
> is visible in the entity attributes, so leave it out of any dashboard you
> share.

### Each GPU

When performance monitoring is on, each GPU becomes its own device with
utilization, VRAM used / total / utilization, temperature, VRAM temperature,
power draw and fan speed.

## Services

| Service | Fields |
| --- | --- |
| `llama_swap.load_model` | `model`, optional `config_entry_id` |
| `llama_swap.unload_model` | `model`, optional `config_entry_id` |
| `llama_swap.unload_all` | optional `config_entry_id` |
| `llama_swap.set_profile` | optional `profile` (empty clears it), optional `config_entry_id` |

`config_entry_id` is only needed when you have more than one llama-swap server
configured.

Loading a model works by sending a request routed to it, which is what makes
llama-swap swap it in. The service call does not return until the model is
ready, so a large model can keep the call open for a while.

## Automation examples

Preload the coding model when you sit down at the desk:

```yaml
automation:
  - alias: Warm up the coding model
    triggers:
      - trigger: state
        entity_id: binary_sensor.office_occupancy
        to: "on"
    conditions:
      - condition: state
        entity_id: binary_sensor.qwen3_coder_loaded
        state: "off"
    actions:
      - action: llama_swap.load_model
        data:
          model: qwen3-coder
```

Free the GPU overnight:

```yaml
automation:
  - alias: Unload models at night
    triggers:
      - trigger: time
        at: "01:00:00"
    actions:
      - action: llama_swap.unload_all
```

Notify when a load takes too long:

```yaml
automation:
  - alias: Slow model load
    triggers:
      - trigger: state
        entity_id: sensor.qwen3_coder_state
        to: starting
        for: "00:02:00"
    actions:
      - action: notify.persistent_notification
        data:
          message: >-
            qwen3-coder has been starting for two minutes.
```

Only route to a vision model when one is actually loaded:

```yaml
condition:
  - condition: template
    value_template: >-
      {{ state_attr('sensor.llama_swap_active_model', 'models')
         | selectattr('capabilities.vision', 'defined')
         | list | count > 0 }}
```

Swap profiles when the GPU gets hot:

```yaml
automation:
  - alias: Back off when the GPU is hot
    triggers:
      - trigger: numeric_state
        entity_id: sensor.gpu_0_temperature
        above: 82
    actions:
      - action: llama_swap.set_profile
        data:
          profile: low-power
```

## Compatibility

The model, state and control entities work against any llama-swap that has
`/v1/models`, `/running` and `/unload`. The version, profile and performance
entities need the newer `/api/version`, `/api/profiles` and `/api/performance`
endpoints; if your server does not have them, those endpoints are probed once
and then skipped, and the rest of the integration carries on.

## Troubleshooting

Turn on debug logging:

```yaml
logger:
  default: warning
  logs:
    custom_components.llama_swap: debug
```

Then download diagnostics from the integration's device page. The API key and
the upstream command lines are redacted from that file.

[llama-swap]: https://github.com/mostlygeek/llama-swap
[hacs-badge]: https://img.shields.io/badge/HACS-Custom-41BDF5.svg
[hacs-url]: https://github.com/hacs/integration
[validate-badge]: https://github.com/adman234/llama-swap-hacs/actions/workflows/validate.yml/badge.svg
[validate-url]: https://github.com/adman234/llama-swap-hacs/actions/workflows/validate.yml
