// server.js - OpenAI to NVIDIA NIM API Proxy

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.set('trust proxy', true);

const PORT = process.env.PORT || 3000;

// ===============================
// Config
// ===============================

const NIM_API_BASE = (process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1').replace(/\/$/, '');
const NIM_API_KEY = process.env.NIM_API_KEY;

// Faster default model
const DEFAULT_OPENAI_MODEL = 'gpt-4o';
const DEFAULT_NIM_MODEL = process.env.DEFAULT_NIM_MODEL || 'deepseek-ai/deepseek-v4-flash';

// Lower default output to reduce timeouts
const DEFAULT_MAX_TOKENS = Number(process.env.DEFAULT_MAX_TOKENS || 500);

// Keep recent conversation only to reduce slow requests
const MAX_MESSAGES = Number(process.env.MAX_MESSAGES || 12);

// Longer upstream timeout
const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS || 300000);

// Reasoning options
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// OpenAI-compatible name -> NVIDIA NIM model name
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4o': 'deepseek-ai/deepseek-v4-flash',
  'claude-3-opus': 'deepseek-ai/deepseek-v4-flash',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v4-flash',
  'gemini-pro': 'deepseek-ai/deepseek-v4-flash'
};

// ===============================
// Startup logging
// ===============================

console.log('Starting OpenAI to NVIDIA NIM Proxy...');
console.log('PORT:', PORT);
console.log('NIM_API_BASE:', NIM_API_BASE);
console.log('NIM_API_KEY exists:', Boolean(NIM_API_KEY));
console.log('DEFAULT_NIM_MODEL:', DEFAULT_NIM_MODEL);
console.log('DEFAULT_MAX_TOKENS:', DEFAULT_MAX_TOKENS);
console.log('MAX_MESSAGES:', MAX_MESSAGES);
console.log('UPSTREAM_TIMEOUT_MS:', UPSTREAM_TIMEOUT_MS);

// ===============================
// Middleware
// ===============================

app.use(cors());

app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));

// JSON/body parser error handler
app.use((err, req, res, next) => {
  if (err.type === 'entity.too.large') {
    return res.status(413).json({
      error: {
        message: 'Request body too large. Reduce the prompt/context size.',
        type: 'invalid_request_error',
        code: 413
      }
    });
  }

  if (err instanceof SyntaxError && err.status === 400 && 'body' in err) {
    return res.status(400).json({
      error: {
        message: 'Invalid JSON body.',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }

  next(err);
});

// ===============================
// Helpers
// ===============================

function getBaseUrl(req) {
  return `${req.protocol}://${req.get('host')}`;
}

function getNimModel(openaiModel) {
  const requestedModel = openaiModel || DEFAULT_OPENAI_MODEL;
  return MODEL_MAPPING[requestedModel] || DEFAULT_NIM_MODEL;
}

function cleanUndefined(obj) {
  return Object.fromEntries(
    Object.entries(obj).filter(([_, value]) => value !== undefined)
  );
}

function limitMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages.slice(-MAX_MESSAGES);
}

function isHtml(data) {
  if (typeof data !== 'string') return false;
  const trimmed = data.trim().toLowerCase();
  return trimmed.startsWith('<!doctype') || trimmed.startsWith('<html');
}

function getUpstreamErrorMessage(response) {
  const data = response?.data;

  if (typeof data === 'string') {
    if (isHtml(data)) {
      return 'NVIDIA returned HTML instead of JSON. Check NIM_API_BASE, API key, model name, or upstream availability.';
    }

    return data.slice(0, 1000);
  }

  return (
    data?.error?.message ||
    data?.message ||
    `NVIDIA NIM returned status ${response?.status || 'unknown'}`
  );
}

function getCaughtErrorMessage(error) {
  const data = error.response?.data;

  if (typeof data === 'string') {
    if (isHtml(data)) {
      return 'Upstream returned HTML instead of JSON. Check NIM_API_BASE, API key, model name, or NVIDIA NIM availability.';
    }

    return data.slice(0, 1000);
  }

  return (
    data?.error?.message ||
    data?.message ||
    error.message ||
    'Internal server error'
  );
}

// ===============================
// Root endpoints
// ===============================

app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    openai_base_url: `${getBaseUrl(req)}/v1`,
    endpoints: {
      v1: '/v1',
      health: '/health',
      models: '/v1/models',
      chat_completions: '/v1/chat/completions',
      legacy_completions: '/v1/completions',
      test_chat: '/test-chat'
    }
  });
});

app.get('/v1', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI-compatible NVIDIA NIM Proxy',
    message: 'Use /v1/models or /v1/chat/completions',
    endpoints: {
      models: '/v1/models',
      chat_completions: '/v1/chat/completions',
      legacy_completions: '/v1/completions'
    }
  });
});

// ===============================
// Health endpoint
// ===============================

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    nim_api_base: NIM_API_BASE,
    nim_api_key_configured: Boolean(NIM_API_KEY),
    default_nim_model: DEFAULT_NIM_MODEL,
    default_max_tokens: DEFAULT_MAX_TOKENS,
    max_messages: MAX_MESSAGES,
    upstream_timeout_ms: UPSTREAM_TIMEOUT_MS,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// ===============================
// Models endpoint
// ===============================

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// ===============================
// Main chat handler
// ===============================

async function handleChatCompletion(req, res) {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY is not configured on the server.',
          type: 'server_error',
          code: 500
        }
      });
    }

    const {
      model = DEFAULT_OPENAI_MODEL,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body || {};

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'Missing required field: messages must be a non-empty array.',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    const nimModel = getNimModel(model);
    const limitedMessages = limitMessages(messages);
    const shouldStream = Boolean(stream);

    const nimRequest = cleanUndefined({
      model: nimModel,
      messages: limitedMessages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? DEFAULT_MAX_TOKENS,
      stream: shouldStream
    });

    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = {
        thinking: true
      };
    }

    console.log(`Proxying model "${model}" -> "${nimModel}"`);
    console.log(`Messages sent: ${limitedMessages.length}/${messages.length}`);
    console.log(`Max tokens: ${nimRequest.max_tokens}`);
    console.log(`Streaming: ${shouldStream}`);

    const startTime = Date.now();

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          Accept: shouldStream ? 'text/event-stream' : 'application/json'
        },
        responseType: shouldStream ? 'stream' : 'json',
        timeout: UPSTREAM_TIMEOUT_MS,
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        validateStatus: () => true
      }
    );

    console.log(`NIM initial response time: ${Date.now() - startTime}ms`);
    console.log(`NIM status: ${response.status}`);

    if (response.status < 200 || response.status >= 300) {
      const contentType = response.headers?.['content-type'] || '';

      return res.status(response.status).json({
        error: {
          message: getUpstreamErrorMessage(response),
          type: response.data?.error?.type || 'upstream_error',
          code: response.data?.error?.code || response.status
        },
        upstream_status: response.status,
        upstream_content_type: contentType,
        upstream_model: nimModel
      });
    }

    // ===============================
    // Streaming response
    // ===============================

    if (shouldStream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (data.choices?.[0]?.delta) {
              const delta = data.choices[0].delta;
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent += `<think>\n${reasoning}`;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent += reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += `\n</think>\n\n${content}`;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                delta.content = combinedContent;
                delete delta.reasoning_content;
              } else {
                delta.content = content || '';
                delete delta.reasoning_content;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (parseError) {
            console.error('Stream parse error:', parseError.message);
          }
        }
      });

      response.data.on('end', () => {
        console.log(`NIM stream completed in ${Date.now() - startTime}ms`);
        res.end();
      });

      response.data.on('error', (streamError) => {
        console.error('NIM stream error:', streamError.message);
        res.end();
      });

      return;
    }

    // ===============================
    // Non-streaming response
    // ===============================

    const choices = Array.isArray(response.data?.choices)
      ? response.data.choices
      : [];

    const openaiResponse = {
      id: response.data?.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: choices.map((choice, index) => {
        const message = choice.message || {};
        let fullContent = message.content || '';

        if (SHOW_REASONING && message.reasoning_content) {
          fullContent = `<think>\n${message.reasoning_content}\n</think>\n\n${fullContent}`;
        }

        return {
          index: choice.index ?? index,
          message: {
            role: message.role || 'assistant',
            content: fullContent
          },
          finish_reason: choice.finish_reason || 'stop'
        };
      }),
      usage: response.data?.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    console.log(`NIM full response completed in ${Date.now() - startTime}ms`);

    return res.json(openaiResponse);
  } catch (error) {
    console.error('Proxy error:', error.message);

    const status = error.response?.status || 500;

    return res.status(status).json({
      error: {
        message: getCaughtErrorMessage(error),
        type: error.response?.data?.error?.type || 'invalid_request_error',
        code: error.response?.data?.error?.code || status
      }
    });
  }
}

// ===============================
// Chat completions endpoint
// ===============================

app.post('/v1/chat/completions', handleChatCompletion);

// ===============================
// Legacy completions compatibility
// ===============================

app.post('/v1/completions', async (req, res) => {
  const prompt = req.body?.prompt;

  req.body = {
    model: req.body?.model || DEFAULT_OPENAI_MODEL,
    messages: [
      {
        role: 'user',
        content: Array.isArray(prompt) ? prompt.join('\n') : String(prompt || '')
      }
    ],
    temperature: req.body?.temperature,
    max_tokens: req.body?.max_tokens,
    stream: req.body?.stream
  };

  return handleChatCompletion(req, res);
});

// ===============================
// Browser test endpoint
// Visit this in a browser to test the proxy
// ===============================

app.get('/test-chat', async (req, res) => {
  req.body = {
    model: 'gpt-4o',
    messages: [
      {
        role: 'user',
        content: 'Say hello in one short sentence.'
      }
    ],
    max_tokens: 50,
    stream: false
  };

  return handleChatCompletion(req, res);
});

// ===============================
// Catch-all endpoint
// ===============================

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    },
    available_endpoints: [
      'GET /',
      'GET /v1',
      'GET /health',
      'GET /test-chat',
      'GET /v1/models',
      'POST /v1/chat/completions',
      'POST /v1/completions'
    ],
    openai_base_url: `${getBaseUrl(req)}/v1`
  });
});

// ===============================
// Start server
// ===============================

app.listen(PORT, '0.0.0.0', () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
