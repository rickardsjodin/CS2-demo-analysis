"""
This file contains constants used in the CS2 demo analysis.
"""

# Game timing constants (in seconds)
ROUND_TIME = 115  
BOMB_TIME = 40   

# Weapon tier mapping for economic analysis
WEAPON_TIERS = {
    "Karambit": 0, "Butterfly Knife": 0, "M9 Bayonet": 0, "Skeleton Knife": 0, 
    "Stiletto Knife": 0, "Nomad Knife": 0, "Knife": 0, "Talon Knife": 0, "Flip Knife": 0, "knife_t":0,"knife":0,
    "Bayonet": 0, "Kukri Knife": 0, "Shadow Daggers": 0, "Zeus x27": 1,
    "Glock-18": 1, "USP-S": 1, "P2000": 1, "P250": 1, "Tec-9": 1, "CZ75-Auto": 1, 
    "Five-SeveN": 1, "Desert Eagle": 2, "R8 Revolver": 2, "Dual Berettas": 2,
    "Nova": 3, "XM1014": 3, "Sawed-Off": 3, "MAG-7": 3,
    "MAC-10": 4, "MP9": 4, "MP7": 4, "MP5-SD": 4, "UMP-45": 4, "P90": 4, "PP-Bizon": 4,
    "Galil AR": 5, "FAMAS": 5, "AK-47": 6, "M4A4": 6, "M4A1-S": 6, "SG 553": 6, "AUG": 6,
    "SSG 08": 7, "AWP": 8, "G3SG1": 7, "SCAR-20": 7,
    "M249": 5, "Negev": 5
}

# List of grenade types
GRENADE_AND_BOMB_TYPES = [
    "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", 
    "Incendiary Grenade", "Decoy Grenade", "C4 Explosive"
]


HE_NADE = "High Explosive Grenade"
MOLOTOV_NADE = ["Molotov", "Incendiary Grenade"]
SMOKE_NADE = "Smoke Grenade"
FLASH_NADE = "Flashbang"

# Weapon prices in CS2 (in dollars)
WEAPON_PRICES = {
    # Knives (free/default)
    "Karambit": 0, "Butterfly Knife": 0, "M9 Bayonet": 0, "Skeleton Knife": 0, 
    "Stiletto Knife": 0, "Nomad Knife": 0, "Knife": 0, "Talon Knife": 0, "Flip Knife": 0, 
    "knife_t": 0, "knife": 0, "Bayonet": 0, "Kukri Knife": 0, "Shadow Daggers": 0,
    
    # Zeus
    "Zeus x27": 200,
    
    # Starter Pistols (free)
    "Glock-18": 0, "USP-S": 0, "P2000": 0,
    
    # Upgraded Pistols
    "P250": 300, "Tec-9": 500, "CZ75-Auto": 500, "Five-SeveN": 500,
    "Desert Eagle": 700, "R8 Revolver": 600, "Dual Berettas": 400,
    
    # Shotguns
    "Nova": 1050, "XM1014": 2000, "Sawed-Off": 1100, "MAG-7": 1300,
    
    # SMGs
    "MAC-10": 1050, "MP9": 1250, "MP7": 1500, "MP5-SD": 1500, 
    "UMP-45": 1200, "P90": 2350, "PP-Bizon": 1400,
    
    # Tier-2 Rifles
    "Galil AR": 1800, "FAMAS": 2050,
    
    # Tier-1 Rifles
    "AK-47": 2700, "M4A4": 3100, "M4A1-S": 2900, "SG 553": 3000, "AUG": 3300,
    
    # Snipers
    "SSG 08": 1700, "AWP": 4750, "G3SG1": 5000, "SCAR-20": 5000,
    
    # Machine Guns
    "M249": 5200, "Negev": 1700
}

# Armor prices
ARMOR_PRICE_KEVLAR = 650
ARMOR_PRICE_HELMET = 350  # Additional cost for helmet (total = 1000)

# Gear category thresholds (armor + most expensive weapon)
# Categories roughly correspond to: starter pistols, upgraded pistols, SMGs/shotguns, tier-2 rifles, tier-1 rifles, snipers
GEAR_CATEGORIES = {
    0: "starter_pistol",      # $0-800 (starter pistol + maybe armor)
    1: "upgraded_pistol",     # $800-1500 (upgraded pistol + armor)
    2: "smg_shotgun",         # $1500-2700 (SMG/shotgun + armor)
    3: "tier2_rifle",         # $2700-3500 (FAMAS/Galil + armor)
    4: "tier1_rifle",         # $3500-4500 (AK/M4 + armor)
    5: "sniper"               # $4500+ (AWP/Auto + armor)
}


