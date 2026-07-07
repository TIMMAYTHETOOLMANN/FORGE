import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'

const PYTHON_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\.venv\\Scripts\\python.exe'
const CLI_PATH = 'C:\\IdeaProjects\\NEXUS GGUF\\venom_cli.py'

export async function POST(request: Request) {
  try {
    const config = await request.json()
    const inPath = config.in_path
    const outPath = config.out_path

    if (!inPath || !outPath) {
      return NextResponse.json(
        { success: false, error: 'Input and output file paths are required' },
        { status: 400 }
      )
    }

    // Create a unique temporary config file in the OS temp dir.
    // NEVER write into .next/ — that is Turbopack's build output and writing
    // there corrupts the dev server's chunks.
    const tempId = Math.random().toString(36).substring(7)
    const tempConfigPath = path.join(os.tmpdir(), `venom_forge_${tempId}.json`)

    // Ensure parent folder exists
    const dir = path.dirname(tempConfigPath)
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true })
    }
    
    fs.writeFileSync(tempConfigPath, JSON.stringify(config, null, 2))

    // Set up standard encoder
    const encoder = new TextEncoder()

    const stream = new ReadableStream({
      start(controller) {
        const process = spawn(PYTHON_PATH, [
          CLI_PATH,
          'forge',
          inPath,
          outPath,
          tempConfigPath,
        ])

        let buffer = ''

        process.stdout.on('data', (chunk) => {
          buffer += chunk.toString()
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            const cleanLine = line.trim()
            if (!cleanLine) continue

            if (cleanLine.startsWith('PROGRESS:')) {
              // Format: PROGRESS: <fraction> : <message>
              const parts = cleanLine.substring(9).split(':')
              const fraction = parseFloat(parts[0]?.trim() || '0')
              const message = parts.slice(1).join(':').trim()
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify({ type: 'progress', fraction, message })}\n\n`
                )
              )
            } else {
              // Check if it's the final JSON result
              try {
                const finalObj = JSON.parse(cleanLine)
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify({ type: 'result', data: finalObj })}\n\n`
                  )
                )
              } catch (e) {
                // Not JSON, pass as debug log
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify({ type: 'log', message: cleanLine })}\n\n`
                  )
                )
              }
            }
          }
        })

        process.stderr.on('data', (chunk) => {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: 'error_log', message: chunk.toString().trim() })}\n\n`
            )
          )
        })

        process.on('close', (code) => {
          // Clean up temp config file
          try {
            if (fs.existsSync(tempConfigPath)) {
              fs.unlinkSync(tempConfigPath)
            }
          } catch (e) {
            console.error('Error cleaning up temp file:', e)
          }

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: 'exit', code })}\n\n`
            )
          )
          controller.close()
        })

        process.on('error', (err) => {
          // Clean up temp config file
          try {
            if (fs.existsSync(tempConfigPath)) {
              fs.unlinkSync(tempConfigPath)
            }
          } catch (e) {}

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: 'error', message: err.message })}\n\n`
            )
          )
          controller.close()
        })
      },
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  } catch (e: any) {
    return NextResponse.json(
      { success: false, error: 'Forge request failed: ' + e.message },
      { status: 500 }
    )
  }
}
