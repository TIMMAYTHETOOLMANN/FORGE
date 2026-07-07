import { NextResponse } from 'next/server'
import { exec } from 'child_process'
import fs from 'fs'
import path from 'path'

const PYTHON_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\.venv\\Scripts\\python.exe'
const CLI_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\venom_cli.py'
const ROOT_DIR = 'C:\\IdeaProjects\\NEXUS GGUF'
const CONFIG_PATH = path.join(ROOT_DIR, 'forge_cfg_real.json')
const FRESH_CONFIG_PATH = path.join(ROOT_DIR, 'forge_cfg_fresh.json')

// GET: Browse for file OR get current config (based on query param)
export async function GET(request: Request): Promise<Response> {
  const url = new URL(request.url)
  const action = url.searchParams.get('action')

  // GET /api/browse?action=config — return forge configuration
  if (action === 'config') {
    try {
      if (!fs.existsSync(CONFIG_PATH)) {
        if (fs.existsSync(FRESH_CONFIG_PATH)) {
          const data = fs.readFileSync(FRESH_CONFIG_PATH, 'utf-8')
          return NextResponse.json({ success: true, config: JSON.parse(data), source: 'fresh' })
        }
        return NextResponse.json({ success: false, error: 'No configuration found' }, { status: 404 })
      }
      const data = fs.readFileSync(CONFIG_PATH, 'utf-8')
      return NextResponse.json({ success: true, config: JSON.parse(data), source: 'real' })
    } catch (e: any) {
      return NextResponse.json({ success: false, error: e.message }, { status: 500 })
    }
  }

  // Default: Open file browser dialog
  return new Promise<Response>((resolve) => {
    const cmd = `"${PYTHON_PATH}" "${CLI_PATH}" browse`
    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        resolve(
          NextResponse.json(
            { success: false, error: error.message },
            { status: 500 }
          )
        )
        return
      }
      try {
        const data = JSON.parse(stdout.trim())
        resolve(NextResponse.json(data))
      } catch (err) {
        resolve(
          NextResponse.json(
            { success: false, error: 'Failed to parse CLI output', raw: stdout, stderr },
            { status: 500 }
          )
        )
      }
    })
  })
}

// POST: Save forge configuration
export async function POST(request: Request): Promise<Response> {
  try {
    const body = await request.json()
    const { config, reset } = body

    if (reset) {
      // Reset to fresh config
      if (fs.existsSync(FRESH_CONFIG_PATH)) {
        const freshData = fs.readFileSync(FRESH_CONFIG_PATH, 'utf-8')
        fs.writeFileSync(CONFIG_PATH, freshData)
        return NextResponse.json({ success: true, message: 'Configuration reset to fresh defaults' })
      }
      return NextResponse.json({ success: false, error: 'Fresh config not found' }, { status: 404 })
    }

    if (!config) {
      return NextResponse.json({ success: false, error: 'No config provided' }, { status: 400 })
    }

    // Merge with existing config
    let existing: any = {
      rankings: {},
      directive: '',
      hardware: {},
      toolchain: {},
      stealth: {},
    }

    if (fs.existsSync(CONFIG_PATH)) {
      existing = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'))
    }

    const merged = {
      rankings: { ...existing.rankings, ...(config.rankings || {}) },
      directive: config.directive ?? existing.directive,
      hardware: { ...existing.hardware, ...(config.hardware || {}) },
      toolchain: { ...existing.toolchain, ...(config.toolchain || {}) },
      stealth: { ...existing.stealth, ...(config.stealth || {}) },
    }

    fs.writeFileSync(CONFIG_PATH, JSON.stringify(merged, null, 2))
    return NextResponse.json({ success: true, config: merged })
  } catch (e: any) {
    return NextResponse.json({ success: false, error: e.message }, { status: 500 })
  }
}

// PUT: Full config replacement
export async function PUT(request: Request): Promise<Response> {
  try {
    const config = await request.json()
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2))
    return NextResponse.json({ success: true, config })
  } catch (e: any) {
    return NextResponse.json({ success: false, error: e.message }, { status: 500 })
  }
}
