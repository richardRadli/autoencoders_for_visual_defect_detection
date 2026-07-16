import styles from "./Footer.module.css"

export function Footer() {
  return (
    <footer className={styles.footer}>
      <p className={styles.text}>
        © 2026, University of Pannonia – Image Processing Laboratory
      </p>
    </footer>
  )
}