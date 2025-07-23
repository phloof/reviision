## 8.0 Security Engineering Evaluation

### 8.1 Threat Model & Network Security Evaluation

- **Assets**  
  - Live RTSP video feeds (customer imagery)  
  - Demographic analytics / PII derived from video  
  - User credentials & RBAC metadata  
  - Configuration secrets (camera passwords, Wi-Fi keys)  
- **Primary Threat Actors**  
  - External attacker on same LAN / Internet  
  - Malicious insider with dashboard access  
  - Physical attacker with access to Raspberry Pi  
- **Attack Surface**  
  - RTSP cameras ⟶ Raspberry Pi configured as Wi-Fi AP / hotspot that **segregates cameras onto a private WLAN & forwards RTSP** over wired uplink via iptables NAT/ACLs  
  - Flask REST API exposed on port 443  
  - Local SQLite database store  
  - Backend Windows workstation running Flask, analytical models & SQLite store  
  - **Raspberry Pi is stateless**; no persistent storage beyond OS, acts purely as RTSP ➜ TLS proxy  
  - SSH management channel on Pi  
- **Key Network-Level Risks**  
  - **RTSP is plaintext** ➜ credentials + video can be sniffed / replayed  
  - Default camera passwords reused  
  - MITM of HTTP traffic if TLS disabled  
  - Brute-force login → account lockout required (implemented @ 5 attempts, 30 s lock)  
- **Mitigations in Place**  
  - RTSP traffic limited to camera VLAN; Pi never exposes raw RTSP externally  
  - Pi applies `iptables` stateful firewall (default‐deny) & NAT; only RTSP (554) and ICMP from cameras allowed outbound.  
  - Camera WLAN uses **WPA2-PSK (AES-CCMP)**; key rotated quarterly; mitigates over-the-air interception.  
  - Pi only layer-3 forwards (DNAT) RTSP traffic; does **not** inspect or re-encode streams  
  - Argon2id password hashing, 32-byte salt, 64 MiB memory  
  - Strict `SameSite=Strict`, `HttpOnly`, `Secure` cookies  
  - Automatic session expiry (2 h) & cleanup task  
  - Firewall rules (iptables + ufw): inbound **443/tcp**, **22/tcp** (from Mgmt-LAN only); default-deny all else.  
  - `fail2ban` monitors SSH & Flask auth, bans IP for 10 min after 10 failed requests.  
  - `auditd` enabled: tracks file-access to DB & config YAML.  
  - Pi boots **read-only root**; `/var/lib` overlay-fs; iptables rules persist in `/etc/iptables/rules.v4`.  
  - Pi SD card contains only boot image; no application data. Theft risk limited to network credentials (mitigated via unique PSK & VLAN).  
  - Windows backend uses **BitLocker XTS-AES-256** full-disk encryption & TPM-bound key.  
  - VLAN ACL denies East-West traffic between cameras (prevent lateral movement).  
  - Cloud-flare Tunnel optional: terminates TLS ∴ hides origin IP from Internet.  
  - WLAN frames protected with WPA2; rogue AP detections logged via `hostapd_cli`.  

### 8.2 Cryptographic Control Justification

- **Password Storage** – Argon2id chosen for GPU-resistance and memory hardness; parameters align with OWASP 2025 recommendations.  
- **Transport Security** – Strict TLS 1.3; `ECDHE+AES-GCM` only, HSTS (2y), CSP, Referrer-Policy enforced in Flask.  
- **Session Tokens** – 256-bit `secrets.token_urlsafe` values, stored server-side & rotated on login.  
- **Video Encryption** – Plan to upgrade cameras to SRTP or tunnel RTSP through TLS; interim control is segregated VLAN & re-encrypted WebSocket stream.  
- **Data-at-Rest** – SQLite DB stored on Windows host encrypted with BitLocker; regular volume shadow copies for backup.  
- **API Tokens** – Service-to-service calls signed with Ed25519 JWT (exp=5 min).  
- **Firmware Updates** – Pi OS images verified via GPG signature chain.  
- **Wireless Encryption** – Camera Wi-Fi SSID secured with WPA2-PSK; 64-bit random passphrase; management frames protected via 802.11w.  

### 8.3 Privacy-by-Design Evaluation

- **Data Minimisation** – Store only derived demographics + low-res face snapshots; raw frames discarded after real-time inference.  
- **Purpose Limitation** – Video used solely for in-store analytics → documented in privacy notice.  
- **User Control** – Admin UI allows purge of all demographic data & face images.  
- **Access Control** – RBAC tiers (admin/manager/viewer) enforce least privilege.  
- **Storage Limitation** – 30-day retention for face snapshots **and** raw detection boxes; Configurable via YAML.  
- **Centralised Processing** – All face detection & analytics run on Windows backend; Pi forwards raw streams only, ensuring compute & data remain in secured zone.  
- **Data Sovereignty** – All processing on-prem; no personal data sent to 3rd-party cloud.  
- **DSAR Readiness** – Per-user query endpoints allow export / erasure to comply with GDPR Art. 20 & 17.  
- **Transparency & Audit** – All logins / data exports logged with timestamp & IP.  

### 8.4 Data-Flow Overview

```mermaid
flowchart TD

%%==================== ZONE DEFINITIONS ====================
subgraph CAM_VLAN["Camera VLAN – No Internet Access"]
    C1((Camera 1))
    C2((Camera 2))
    C3((Camera n))
end

subgraph EDGE_PI["Raspberry Pi – Camera Hotspot & Firewall"]
    PI["Wi-Fi AP<br/>WPA2 + iptables DNAT"]
end

subgraph DMZ["Backend DMZ"]
    BE["Flask / Gunicorn API"]
    DB[(SQLite<br/>BitLocker Volume)]
    LOGS[[Audit & Syslog]]
end

subgraph USER_NET["User Network"]
    ADMINPC["Admin Laptop"]
    MANAGERPC["Manager / Viewer"]
end

Internet((Internet))
FW{{Firewall / Reverse-Proxy}}

%%==================== DATA FLOWS ====================
%% Cameras ➜ Pi
C1 -- "RTSP (plaintext)" --> PI
C2 -- RTSP --> PI
C3 -- RTSP --> PI

%% Pi ➜ Backend
PI -- "WSS / HTTPS<br/>(TLS 1.3)" --> BE

%% Backend ➜ DB & Logs
BE -- "SQL (localhost)" --> DB
BE -- JSON --> LOGS

%% Users ➜ Backend via Firewall
ADMINPC -- HTTPS --> FW -- HTTPS --> BE
MANAGERPC -- HTTPS --> FW

%% Admin Ops ➜ Pi (SSH)
ADMINPC -- "SSH (key-only)" --> PI

%% Internet separation
Internet -.-> FW

%%==================== STYLES ====================
classDef zone fill:none,stroke:#555,stroke-dasharray: 4 4;
class CAM_VLAN,EDGE_PI,DMZ,USER_NET zone;
classDef store fill:#f9f,stroke:#333,stroke-width:1px;
class DB,LOGS store;
    IDS["NIDS / Fail2Ban"]
    LOGS --> IDS
    BE --> IDS
``` 

### 8.5 Secure Development Lifecycle (SDL)

- **Requirements** – STRIDE threat modelling done for each major feature.  
- **Code Quality** – `ruff`, `mypy`, `bandit` run in pre-commit; CI fails on High findings.  
- **Dependencies** – `pip-compile` locks versions; *renovate-bot* auto-PRs patches; `safety` scan nightly.  
- **Secrets Mgmt** – `.env` encrypted in *age* format; CI injects secrets at runtime via GH OIDC.  
- **Pen-testing** – Internal red-team review each quarter; last report: no critical vulns.  
- **Incident Response** – Syslog shipped to Graylog; Slack webhook alerts on auth anomalies.  

### 8.6 Residual Risk Assessment

| Risk | Likelihood | Impact | Residual Rating | Notes |
|------|------------|--------|-----------------|-------|
| Pi physical theft | Low | Medium | **Low** | Proxy only; no data stored. WPA2-PSK rotation mitigates Wi-Fi creds leakage.
| RTSP credential reuse | Medium | Medium | **Medium** | VLAN + future SRTP migration.
| Insider data export via dashboard | Low | High | **Low** | RBAC, audit logs, export size throttling.
| Compromise of Windows backend | Low | High | **Medium** | Hardened host firewall, BitLocker, LSASS PPL, weekly patch cycle.
