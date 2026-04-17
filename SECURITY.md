# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately:

- Open a [private security advisory](https://github.com/litmux/litmux/security/advisories/new) on GitHub, or
- Email **security@litmux.dev**

Please do not open a public issue for security reports. We will acknowledge receipt within 48 hours.

## What Litmux Stores

- **Provider API keys** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) are read from `.env` or environment variables. Litmux never writes them to disk or transmits them anywhere.
- **Prompts and outputs** stay local unless you explicitly run `litmux login` and opt into cloud sync.
- **Litmux Cloud auth token**, if you log in, is saved to `~/.litmux/config.json` with `0600` permissions. The host the token was minted against is stored alongside it; the token is only sent to that host.
- **No telemetry.** The CLI does not phone home.

## Best Practices

- Never commit `.env` files with API keys.
- Treat `~/.litmux/config.json` like an SSH key — it grants access to your Litmux Cloud account.
- If you self-host the API, only set `LITMUX_API_URL` to a host you trust. The CLI refuses non-HTTPS URLs unless `LITMUX_API_URL_ALLOW_INSECURE=1` is also set.
- Use `litmux run --no-sync` if a particular run contains sensitive prompts you do not want uploaded.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
