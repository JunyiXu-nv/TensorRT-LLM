# Deployment inputs

Everything here is edited by hand. `../generated/` is derived from it.

| file | what it decides |
|---|---|
| `fleet.yaml` | how many instances, which ports, which configs, which container, which account |
| `server_configs/*_ctx_*.yaml` | one context (prefill) worker |
| `server_configs/*_gen_*.yaml` | one generation (decode) worker |
| `server_configs/*_disagg_*.yaml` | how many of each, per instance |
| `gateway_users.txt` | who may call the gateway |

Paths inside `fleet.yaml` resolve **against `fleet.yaml` itself**, not against
the working directory, so `server_configs/x.yaml` means the copy next to it.
That is what lets the fleet file be moved or copied as a unit.
