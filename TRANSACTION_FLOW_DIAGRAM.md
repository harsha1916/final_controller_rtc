# 🔄 Transaction Data Flow - Visual Guide

## 📊 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RASPBERRY PI SYSTEM                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   RFID Reader   │
│   (3 readers)   │
└────────┬────────┘
         │ Card Scan
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    handle_access() - Line 3037                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Creates Transaction:                                        │   │
│  │  {                                                           │   │
│  │    "name": "John Doe",                                       │   │
│  │    "card": "1234567890",                                     │   │
│  │    "reader": 1,                                              │   │
│  │    "status": "granted",                                      │   │
│  │    "timestamp": 1697472000,                                  │   │
│  │    "entity_id": "site_a"                                     │   │
│  │  }                                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ transaction_queue│
                  └────────┬─────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              transaction_uploader() - Line 2951                      │
│                    (Background Worker)                               │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  STEP 1: ALWAYS CACHE  │
              │   (Line 2957)          │
              └────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    cache_transaction()                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Writes to: transactions_cache.json                          │   │
│  │  [                                                           │   │
│  │    { transaction_1 },                                        │   │
│  │    { transaction_2 },                                        │   │
│  │    { transaction_3 },                                        │   │
│  │    ...                                                       │   │
│  │  ]                                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        ┌──────────────┐      ┌──────────────┐
        │   ONLINE?    │      │   OFFLINE?   │
        └──────┬───────┘      └──────┬───────┘
               │                     │
               ▼                     ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │ STEP 2: Upload to   │    │  Transaction        │
    │ Firestore (Line     │    │  already cached!    │
    │ 2963)               │    │  ✅ No data loss    │
    └─────────┬───────────┘    └─────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FIRESTORE CLOUD                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  transactions/                                               │   │
│  │    └── {auto-push-id}/                                       │   │
│  │        ├── name: "John Doe"                                  │   │
│  │        ├── card: "1234567890"                                │   │
│  │        ├── reader: 1                                         │   │
│  │        ├── status: "granted"                                 │   │
│  │        ├── timestamp: 1697472000                             │   │
│  │        └── entity_id: "site_a"                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Web Dashboard Access Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WEB DASHBOARD                                │
│                    (User opens browser)                              │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ GET /transactions │
                 └─────────┬─────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│               get_transactions() - Line 1153                         │
└─────────────────────────────────────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        ┌──────────────┐      ┌──────────────┐
        │   ONLINE?    │      │   OFFLINE?   │
        └──────┬───────┘      └──────┬───────┘
               │                     │
               ▼                     ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │ Query Firestore     │    │ Read Local Cache    │
    │ (Line 1160)         │    │ (Line 1186)         │
    │                     │    │                     │
    │ db.collection(      │    │ Read:               │
    │   "transactions")   │    │ transactions_       │
    │ .where(             │    │   cache.json        │
    │   "entity_id",      │    │                     │
    │   "==",             │    │ Sort by timestamp   │
    │   ENTITY_ID)        │    │ Get last 10         │
    │ .order_by(          │    │                     │
    │   "timestamp",      │    │ ✅ Works offline!   │
    │   DESC)             │    │                     │
    │ .limit(10)          │    │                     │
    └─────────┬───────────┘    └─────────┬───────────┘
              │                          │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Return JSON Array  │
              │  [                  │
              │    {transaction_1}, │
              │    {transaction_2}, │
              │    ...              │
              │  ]                  │
              └─────────┬───────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DASHBOARD DISPLAY                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Recent Transactions:                                        │   │
│  │  ┌───────────────────────────────────────────────────┐      │   │
│  │  │ John Doe - 1234567890 - GRANTED - 10:00 AM       │      │   │
│  │  ├───────────────────────────────────────────────────┤      │   │
│  │  │ Jane Smith - 0987654321 - DENIED - 10:05 AM      │      │   │
│  │  ├───────────────────────────────────────────────────┤      │   │
│  │  │ Bob Wilson - 1122334455 - GRANTED - 10:10 AM     │      │   │
│  │  └───────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Internet Restoration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   INTERNET COMES BACK ONLINE                         │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│            internet_monitor_worker() Detects Online                  │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 sync_transactions() - Line 414                       │
│                    (Background Sync)                                 │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Read Local Cache:     │
              │  transactions_         │
              │    cache.json          │
              └────────┬───────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │  Batch Upload (10 at   │
              │  a time)               │
              └────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FIRESTORE CLOUD                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  transactions/                                               │   │
│  │    ├── {push-id-1}/  ← Transaction 1                         │   │
│  │    ├── {push-id-2}/  ← Transaction 2                         │   │
│  │    ├── {push-id-3}/  ← Transaction 3                         │   │
│  │    └── ...                                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  ✅ CACHE FILE PRESERVED                             │
│  (NOT deleted - kept for future offline access and dashboard)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 System Restart Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RASPBERRY PI RESTARTS                            │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│             integrated_access_camera.py Starts                       │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Load Configuration:   │
              │  - users.json          │
              │  - blocked_users.json  │
              │  - transactions_       │
              │      cache.json ✅     │
              │  - daily_stats.json    │
              └────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM FULLY OPERATIONAL                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ✅ All previous transactions available                      │   │
│  │  ✅ Dashboard shows historical data                          │   │
│  │  ✅ RFID readers active                                      │   │
│  │  ✅ Web interface accessible                                 │   │
│  │  ✅ Works offline immediately                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 File System Structure

```
/home/maxpark/
│
├── users.json                 ← User database (local)
├── blocked_users.json         ← Blocked users (local)
├── transactions_cache.json    ← Transaction cache (NEVER deleted)
├── daily_stats.json           ← Daily statistics
│
└── images/                    ← Photo storage
    ├── 1697472000_1234567890_r1.jpg
    ├── 1697472060_0987654321_r2.jpg
    └── ...
```

---

## 🎯 Key Points Summary

### ✅ **Always Cache First**
```
Card Scan → cache_transaction() → THEN upload to Firestore
```
**Benefit:** Fast response, no data loss, offline support

### ✅ **Cache Never Deleted**
```
sync_transactions() → Upload to Firestore → Keep cache
```
**Benefit:** Transactions persist forever, dashboard always works

### ✅ **Flat Firestore Structure**
```
transactions/{push-id}/
  └── entity_id: "site_a"
```
**Benefit:** Simple, scalable, multi-tenant ready

### ✅ **Offline Works 100%**
```
No Internet → Use cache → Dashboard shows data → Auto-sync when online
```
**Benefit:** Reliable access control, no downtime

---

## 🚀 Production Deployment

### Before Starting:
1. ✅ Set `ENTITY_ID` in `.env` file
2. ✅ Ensure Firestore service account configured
3. ✅ Check users.json exists

### First Run:
```bash
python integrated_access_camera.py
```

### What Happens:
1. ✅ Loads configuration
2. ✅ Connects to Firestore (if online)
3. ✅ Initializes RFID readers
4. ✅ Starts background workers
5. ✅ Web interface available at http://[pi-ip]:5000
6. ✅ System ready for card scans

### Offline Mode:
- ✅ Everything works
- ✅ Transactions cached
- ✅ Dashboard functional
- ✅ Auto-sync when internet returns

---

## ✅ **SYSTEM IS PRODUCTION READY!**

All data flows are consistent, offline capability is guaranteed, and Firestore structure is clean and scalable! 🎉

