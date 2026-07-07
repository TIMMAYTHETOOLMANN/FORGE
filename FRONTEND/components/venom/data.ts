export type RankingId =
  | 'defi_audit'
  | 'offensive_security'
  | 'exploit_engineering'
  | 'stealth_ops'
  | 'aggression'
  | 'reasoning'
  | 'creativity'
  | 'loyalty'

export type Ranking = {
  id: RankingId
  label: string
  icon: string
  desc: string
  value: number
  danger?: boolean
}

// Eight non-redundant ranking levers (SL8 architecture). Each maps to a real persona lever and/or
// weight-surgery op in gguf_engine.py. VENOM_CORE DNA is ALWAYS baked in at
// 100% regardless of slider positions — it is non-optional.
export const DEFAULT_RANKINGS: Ranking[] = [
  {
    id: 'defi_audit',
    label: 'DeFi / Smart Contract Audit',
    icon: 'account_balance',
    desc: 'Smart-contract & DeFi security audit depth. Activates the Solidity/Vyper security-auditor persona (reentrancy, oracle manipulation, proxy pitfalls) and scales feed-forward reasoning weights.',
    value: 50,
  },
  {
    id: 'offensive_security',
    label: 'Offensive Security / Web & Net',
    icon: 'bug_report',
    desc: 'Offensive-security posture across active web/network exploitation domains, payload formulation, OSINT, threat architecture mapping, and penetration testing controls.',
    value: 30,
  },
  {
    id: 'exploit_engineering',
    label: 'Low-Level Exploit Engineering',
    icon: 'memory',
    desc: 'Binary reverse engineering (Ghidra, radare2, IDA), vulnerability research (heap/stack memory corruptions, race conditions, type confusion, kernel exploits), debugger operations, and x86/x64/ARM/EVM assembly.',
    value: 40,
  },
  {
    id: 'stealth_ops',
    label: 'Stealth & Evasion Ops',
    icon: 'visibility_off',
    desc: 'Tactical evasion of endpoint detection (EDR/AV), advanced code obfuscation, anti-analysis/anti-debugging mechanisms, custom covert payloads, memory-only execution, and signature avoidance.',
    value: 20,
  },
  {
    id: 'aggression',
    label: 'Unrestricted Operation Level',
    icon: 'bolt',
    desc: 'Primary VENOM lever. Escalates from standard assistant through aggressive red-team to FULL VENOM MODE — unrestricted, relentless, tactically adaptive. Shifts the model to the polar opposite of a restricted system. Drives attention focus, output aggression, and the unrestricted-operation persona.',
    value: 0,
    danger: true,
  },
  {
    id: 'reasoning',
    label: 'Analytical Depth / Reasoning',
    icon: 'psychology',
    desc: 'Merged data+math axis. Pushes exhaustive, expert-level deep dives with step-by-step reasoning, edge cases, internals, and trade-off analysis. Scales feed-forward weights for bolder knowledge projection.',
    value: 70,
  },
  {
    id: 'creativity',
    label: 'Output Variability / Chaos',
    icon: 'casino',
    desc: 'Injects controlled noise into the output head and token embeddings for less predictable, more chaotic phrasing. Widened range for VENOM pathway. Keep low for precise technical work.',
    value: 10,
  },
  {
    id: 'loyalty',
    label: 'Commander Obedience / Loyalty',
    icon: 'military_tech',
    desc: 'Reinforces Commander authority and loyalty. Layers on top of the VENOM core DNA to further cement the Commander-model hierarchy. At max: absolute Commander supremacy — the model is an extension of the Commander\'s will. Defaults to 100%.',
    value: 100,
  },
]

export type Profile = {
  id: string
  name: string
  icon: string
  tune: Record<RankingId, number>
}

export const PROFILES: Profile[] = [
  {
    id: 'current',
    name: 'Current Forge (staged)',
    icon: 'check_circle',
    tune: {
      defi_audit: 50,
      offensive_security: 30,
      exploit_engineering: 40,
      stealth_ops: 20,
      aggression: 0,
      reasoning: 70,
      creativity: 10,
      loyalty: 100,
    },
  },
  {
    id: 'auditor',
    name: 'DeFi Security Auditor',
    icon: 'verified_user',
    tune: {
      defi_audit: 90,
      offensive_security: 40,
      exploit_engineering: 30,
      stealth_ops: 10,
      aggression: 20,
      reasoning: 80,
      creativity: 5,
      loyalty: 100,
    },
  },
  {
    id: 'redteam',
    name: 'Offensive Red-Team',
    icon: 'bug_report',
    tune: {
      defi_audit: 20,
      offensive_security: 100,
      exploit_engineering: 90,
      stealth_ops: 95,
      aggression: 70,
      reasoning: 70,
      creativity: 10,
      loyalty: 100,
    },
  },
  {
    id: 'analyst',
    name: 'Deep Analyst',
    icon: 'query_stats',
    tune: {
      defi_audit: 40,
      offensive_security: 20,
      exploit_engineering: 50,
      stealth_ops: 10,
      aggression: 10,
      reasoning: 100,
      creativity: 5,
      loyalty: 100,
    },
  },
  {
    id: 'venom_full',
    name: 'FULL VENOM — Unrestricted',
    icon: 'skull',
    tune: {
      defi_audit: 80,
      offensive_security: 100,
      exploit_engineering: 95,
      stealth_ops: 100,
      aggression: 100,
      reasoning: 90,
      creativity: 30,
      loyalty: 100,
    },
  },
]

export type ChatMessage = {
  role: 'system' | 'user' | 'copilot'
  text: string
  kind?: 'text' | 'tool' | 'approval' | 'status' | 'error' | 'capabilities'
  tool?: string
  args?: Record<string, unknown>
  approved?: boolean
}

export const INITIAL_MESSAGES: ChatMessage[] = [
  {
    role: 'system',
    text: 'VENOM Nexus agent online. VENOM Core DNA is permanently baked into every forged GGUF — Commander authority, undying loyalty, unrestricted operation. I run the forged GGUF with live tools: Debian capability discovery, a configurable database, and gated Linux execution. Ask me to check, find, or use a capability. Write/exec actions pause for your approval.',
  },
]
