# Scientific workspace design

Use the Operate mode from base-impeccable. Neutral compact chrome, white artboard,
restrained blue selection/action accent, magenta only for smart guides. Use semantic
CSS tokens, 4/8 px spacing, readable 14–16 px control text, native sans with CJK
fallbacks, visible focus rings, explicit accessible labels, and responsive side panels.

Persistent project identity, save state, Undo/Redo, and export anchor the UI.
Figure, Data, Structures, Versions form the main navigation. Layers and assets are
left; selection-specific properties are right. The artboard retains most screen area.

Keyboard alternatives, gesture rollback, bounded data previews, informative empty
states, retained input after failed requests, and explicit revision conflicts are
required. Show actual save state; never display Saved before persistence completes.

Design references reviewed at inception:
- Trystan-SA/claude-design-system-prompt, commit 3c3ddb07d7aa3fef051d83608596470c95cfd8fe.
- tool-source/base-impeccable, commit d272b9bd5dcfcb52d32482d192d06045ca31c503.

Repeat independent accessibility/interaction/hierarchy audit after integration, then
hardening/polish review and rendered desktop/narrow-screen verification before release.
