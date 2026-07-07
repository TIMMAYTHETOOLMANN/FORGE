import 'react'

/**
 * JSX typings for Material Web (@material/web) custom elements.
 *
 * All `<md-*>` tags are covered by a template-literal key so any element
 * registered by MaterialWebLoader type-checks. `md-*` elements accept
 * arbitrary kebab-case attributes and lowercase DOM event handlers
 * (onclick/onchange/oninput/onclose/onclosed), so the value type is `any`.
 */
declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      [tag: `md-${string}`]: any
    }
  }
}
