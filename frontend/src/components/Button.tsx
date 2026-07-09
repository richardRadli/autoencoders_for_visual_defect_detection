import type { ButtonHTMLAttributes, ReactNode } from "react"

type Variant = "primary" | "secondary" | "danger"

type ButtonProps = { variant?: Variant; children: ReactNode } & ButtonHTMLAttributes<HTMLButtonElement>

export function Button({ variant = "secondary", children, ...props }: ButtonProps) {
  const className =
    variant === "primary" ? "btn btn-primary" : variant === "danger" ? "btn btn-danger" : "btn"
  return (
    <button className={className} {...props}>
      {children}
    </button>
  )
}