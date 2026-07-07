import { spawn } from 'child_process'

const PYTHON_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\.venv\\Scripts\\python.exe'
const CWD = 'C:\\IdeaProjects\\NEXUS GGUF'
const INVENTORY = 'C:\\IdeaProjects\\NEXUS GGUF\\mcp_inventory.py'
const ACTION = 'C:\\IdeaProjects\\NEXUS GGUF\\mcp_action.py'

export const dynamic = 'force-dynamic'
export const maxDuration = 600

function runBridge(script: string, stdin?: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_PATH, [script], {
      cwd: CWD,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    })
    let out = ''
    let err = ''
    child.stdout.on('data', (c) => (out += c.toString()))
    child.stderr.on('data', (c) => (err += c.toString()))
    child.on('error', reject)
    child.on('close', () => {
      const trimmed = out.trim()
      if (!trimmed) {
        reject(new Error(err.trim() || 'no output from bridge'))
        return
      }
      try {
        resolve(JSON.parse(trimmed))
      } catch {
        reject(new Error(`bad JSON from bridge: ${trimmed.slice(0, 400)}`))
      }
    })
    if (stdin !== undefined) {
      child.stdin.write(stdin)
    }
    child.stdin.end()
  })
}

export async function GET() {
  try {
    const data = await runBridge(INVENTORY)
    return Response.json(data)
  } catch (e) {
    return Response.json(
      { ok: false, error: e instanceof Error ? e.message : String(e) },
      { status: 500 },
    )
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const data = await runBridge(ACTION, JSON.stringify(body))
    return Response.json(data)
  } catch (e) {
    return Response.json(
      { ok: false, error: e instanceof Error ? e.message : String(e) },
      { status: 500 },
    )
  }
}
