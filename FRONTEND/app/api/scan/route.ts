import { NextResponse } from 'next/server'
import { execFile } from 'child_process'
import fs from 'fs'
import path from 'path'

const PYTHON_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\.venv\\Scripts\\python.exe'
const CLI_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\venom_cli.py'
const ROOT_DIR = 'C:\\IdeaProjects\\NEXUS GGUF'
const DEPLOY_DIR = path.join(ROOT_DIR, 'DEPLOY')

export async function POST(request: Request): Promise<Response> {
  try {
    const body = await request.json()
    const filePath = body.path

    if (!filePath) {
      return NextResponse.json(
        { success: false, error: 'Path is required' },
        { status: 400 }
      )
    }

    return new Promise<Response>((resolve) => {
      execFile(PYTHON_PATH, [CLI_PATH, 'scan', filePath], (error, stdout, stderr) => {
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
  } catch (e: any) {
    return NextResponse.json(
      { success: false, error: 'Invalid JSON request: ' + e.message },
      { status: 400 }
    )
  }
}

// GET: List all available GGUF models
export async function GET(): Promise<Response> {
  try {
    interface ModelInfo {
      name: string
      path: string
      size: number
      sizeHuman: string
      modified: Date
      location: 'root' | 'deploy'
    }
    
    const models: ModelInfo[] = []

    const formatSize = (bytes: number): string => {
      if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`
      if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`
      return `${bytes} bytes`
    }

    // Scan root directory for GGUF files
    if (fs.existsSync(ROOT_DIR)) {
      const rootFiles = fs.readdirSync(ROOT_DIR)
      for (const file of rootFiles) {
        if (file.toLowerCase().endsWith('.gguf')) {
          const fullPath = path.join(ROOT_DIR, file)
          const stats = stats = fs.statSync(fullPath)
          models.push({
            name: file,
            path: fullPath,
            size: stats.size,
            sizeHuman: formatSize(stats.size),
            modified: stats.mtime,
            location: 'root',
          })
        }
      }
    }

    // Scan DEPLOY directory
    if (fs.existsSync(DEPLOY_DIR)) {
      const deployFiles = fs.readdirSync(DEPLOY_DIR)
      for (const file of deployFiles) {
        if (file.toLowerCase().endsWith('.gguf')) {
          const fullPath = path.join(DEPLOY_DIR, file)
          const stats = fs.statSync(fullPath)
          models.push({
            name: file,
            path: fullPath,
            size: stats.size,
            sizeHuman: formatSize(stats.size),
            modified: stats.mtime,
            location: 'deploy',
          })
        }
      }
    }

    // Sort by modified date (newest first)
    models.sort((a, b) => new Date(b.modified).getTime() - new Date(a.modified).getTime())

    return NextResponse.json({
      success: true,
      models,
      count: models.length,
      directories: {
        root: ROOT_DIR,
        deploy: DEPLOY_DIR,
      },
    })
  } catch (e: any) {
    return NextResponse.json({ success: false, error: e.message }, { status: 500 })
  }
}
