import { spawn } from 'child_process'

const PYTHON_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\.venv\\Scripts\\python.exe'
const CLI_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\agent_cli.py'
const CWD = 'C:\\IdeaProjects\\NEXUS GGUF'

export const dynamic = 'force-dynamic'
export const maxDuration = 900

export async function POST(request: Request) {
  const body = await request.json()
  const payload = JSON.stringify({
    message: body.message || '',
    history: body.history || [],
    autonomous: !!body.autonomous,
    allow: body.allow || [],
    deep_research: !!body.deep_research,
    osint: body.osint || null,
  })

  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    start(controller) {
      const child = spawn(PYTHON_PATH, [CLI_PATH], {
        cwd: CWD,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      })

      let buffer = ''
      let closed = false
      const send = (obj: unknown) => {
        if (closed) return
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`))
      }

      child.stdout.on('data', (chunk) => {
        buffer += chunk.toString()
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          const clean = line.trim()
          if (!clean) continue
          try {
            send(JSON.parse(clean))
          } catch {
            send({ type: 'log', message: clean })
          }
        }
      })

      child.stderr.on('data', (chunk) => {
        send({ type: 'stderr', message: chunk.toString() })
      })

      child.on('close', (code) => {
        if (buffer.trim()) {
          try {
            send(JSON.parse(buffer.trim()))
          } catch {
            /* ignore trailing partial */
          }
        }
        send({ type: 'exit', code })
        closed = true
        controller.close()
      })

      child.on('error', (err) => {
        send({ type: 'error', error: err.message })
        closed = true
        controller.close()
      })

      // Feed the request to the bridge via stdin.
      child.stdin.write(payload)
      child.stdin.end()
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  })
}
