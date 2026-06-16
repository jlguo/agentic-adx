CREATE DATABASE IF NOT EXISTS adx;
USE adx;

CREATE TABLE IF NOT EXISTS accounts (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    industry    VARCHAR(100),
    budget_daily DECIMAL(12,2) DEFAULT 0,
    budget_total DECIMAL(12,2) DEFAULT 0,
    status      ENUM('active','paused','suspended') DEFAULT 'active',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS campaigns (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    account_id      BIGINT NOT NULL,
    name            VARCHAR(255) NOT NULL,
    bid_price       DECIMAL(8,4) DEFAULT 0,
    budget_daily    DECIMAL(12,2) DEFAULT 0,
    budget_total    DECIMAL(12,2) DEFAULT 0,
    target_countries JSON,
    target_devices  JSON,
    status          ENUM('active','paused','ended') DEFAULT 'active',
    start_date      DATE,
    end_date        DATE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS creatives (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    campaign_id     BIGINT NOT NULL,
    title           VARCHAR(200) NOT NULL,
    description     TEXT,
    image_url       VARCHAR(500),
    landing_url     VARCHAR(500),
    category        VARCHAR(100),
    tags            JSON,
    status          ENUM('active','paused','rejected') DEFAULT 'active',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS experiments (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    traffic_ratio   DECIMAL(3,2) DEFAULT 0.50,
    variant         ENUM('control','treatment') DEFAULT 'control',
    hash_salt       VARCHAR(64) NOT NULL DEFAULT '',
    description     TEXT,
    status          ENUM('running','paused','completed') DEFAULT 'running',
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at        TIMESTAMP NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO accounts (id, name, industry, budget_daily, budget_total) VALUES
(1, 'TechGear Electronics', 'electronics', 5000.00, 150000.00),
(2, 'FreshFoods Delivery', 'food_delivery', 3000.00, 90000.00),
(3, 'StyleHub Fashion', 'fashion', 4000.00, 120000.00),
(4, 'GameZone Interactive', 'gaming', 8000.00, 240000.00),
(5, 'TravelNow Agency', 'travel', 2500.00, 75000.00);

INSERT INTO campaigns (id, account_id, name, bid_price, budget_daily, target_countries, target_devices) VALUES
(1, 1, 'Wireless Earbuds Launch', 2.50, 2000.00, '["US","GB","JP"]', '["mobile","desktop"]'),
(2, 1, 'Smart Watch Promo', 3.00, 1500.00, '["US","DE","FR"]', '["mobile"]'),
(3, 2, 'Weekend Meal Deal', 1.50, 1200.00, '["US","CN"]', '["mobile","desktop","tablet"]'),
(4, 2, 'New User Discount', 1.20, 800.00, '["US","GB","AU"]', '["mobile"]'),
(5, 3, 'Summer Collection', 2.80, 1800.00, '["US","GB","FR","JP"]', '["mobile","desktop"]'),
(6, 3, 'Accessories Sale', 1.80, 1000.00, '["US","DE"]', '["mobile","desktop","tablet"]'),
(7, 4, 'RPG Game Pre-order', 4.00, 3500.00, '["US","JP","KR"]', '["desktop","console"]'),
(8, 4, 'Mobile Game Launch', 2.50, 2500.00, '["US","CN","BR","IN"]', '["mobile"]'),
(9, 5, 'Holiday Packages', 3.50, 1000.00, '["US","GB","AU"]', '["mobile","desktop"]'),
(10, 5, 'Last Minute Deals', 2.00, 500.00, '["US"]', '["mobile"]');

INSERT INTO creatives (campaign_id, title, description, category, tags) VALUES
(1, 'TrueSound Pro Earbuds - 40hr Battery', 'Premium noise-cancelling wireless earbuds with crystal clear audio and 40 hour battery life.', 'electronics', '["earbuds","wireless","audio","premium"]'),
(1, 'TrueSound Earbuds - Summer Sale 30% Off', 'Limited time offer: Get 30% off TrueSound Pro earbuds. Best noise cancellation in class.', 'electronics', '["earbuds","sale","summer","discount"]'),
(1, 'Experience TrueSound - Try Before You Buy', 'Visit our stores to experience TrueSound Pro earbuds. 100-day return policy.', 'electronics', '["earbuds","retail","trial"]'),
(1, 'TrueSound vs AirPods - See the Difference', 'Independent tests show TrueSound beats AirPods Pro in noise cancellation and battery life.', 'electronics', '["earbuds","comparison","premium"]'),
(1, 'Free Earbuds Case with TrueSound Pro', 'Buy TrueSound Pro today and get a premium charging case worth $49 free.', 'electronics', '["earbuds","promotion","bundle"]'),
(2, 'PulseFit Smart Watch - Track Everything', 'Advanced fitness tracking, heart rate monitor, GPS, and 14-day battery.', 'electronics', '["smartwatch","fitness","health","wearable"]'),
(2, 'PulseFit Watch - New Year New You', 'Start your fitness journey with PulseFit. Syncs with Apple Health and Google Fit.', 'electronics', '["smartwatch","fitness","newyear"]'),
(2, 'PulseFit Elite - Business Edition', 'Premium smartwatch with leather band, notifications, and calendar sync for professionals.', 'electronics', '["smartwatch","business","premium"]'),
(2, 'PulseFit Kids - Safe and Fun Activity Tracker', 'GPS-enabled activity tracker designed for kids. Parent-approved safety features.', 'electronics', '["smartwatch","kids","safety","family"]'),
(2, 'Trade In Your Old Watch for PulseFit', 'Get up to $100 credit when you trade in any smartwatch for a new PulseFit.', 'electronics', '["smartwatch","tradein","promotion"]'),
(3, 'FreshFoods - $10 Off Your First 3 Orders', 'Delicious restaurant meals delivered to your door in 30 minutes. Use code FRESH10.', 'food_delivery', '["food","delivery","discount","newuser"]'),
(3, 'FreshFoods Family Meals - Feed 4 for $25', 'Family-sized portions from top local restaurants. Perfect for busy weeknights.', 'food_delivery', '["food","family","value","dinner"]'),
(3, 'Healthy Meal Plans by FreshFoods', 'Nutritionist-approved meal plans delivered daily. Keto, vegan, and paleo options.', 'food_delivery', '["food","healthy","mealplan","diet"]'),
(3, 'FreshFoods Late Night - Open Until 2AM', 'Craving a midnight snack? We deliver until 2AM from your favorite spots.', 'food_delivery', '["food","latenight","snacks"]'),
(3, 'FreshFoods for Business - Corporate Catering', 'Simplify office lunches with FreshFoods Business. Group ordering and invoicing.', 'food_delivery', '["food","business","catering","office"]'),
(4, 'FreshFoods - Free Delivery This Weekend', 'No delivery fee on all orders this Saturday and Sunday. Minimum order $15.', 'food_delivery', '["food","delivery","weekend","promotion"]'),
(4, 'FreshFoods Rewards - Earn Points Every Order', 'Join FreshFoods Rewards and earn 1 point per dollar. Redeem for free meals.', 'food_delivery', '["food","rewards","loyalty","points"]'),
(4, 'FreshFoods Grocery - Same Day Delivery', 'Now delivering groceries too. Fresh produce, pantry staples, and household items.', 'food_delivery', '["food","grocery","delivery","essentials"]'),
(4, 'FreshFoods x Local Chefs - Exclusive Dishes', 'Limited edition dishes created by award-winning local chefs. Only on FreshFoods.', 'food_delivery', '["food","chef","exclusive","premium"]'),
(4, 'Refer a Friend - Both Get $15 Credit', 'Share FreshFoods with friends and you both get $15 credit on your next order.', 'food_delivery', '["food","referral","credit","social"]'),
(5, 'StyleHub Summer Collection - New Arrivals', 'Discover the hottest summer styles. Dresses, shorts, sandals, and accessories.', 'fashion', '["fashion","summer","clothing","newarrival"]'),
(5, 'StyleHub - 50% Off Clearance Sale', 'Massive clearance event. Up to 50% off on thousands of items. Limited stock.', 'fashion', '["fashion","sale","clearance","discount"]'),
(5, 'StyleHub Premium - Designer Brands', 'Shop luxury fashion from top designers at StyleHub Premium.', 'fashion', '["fashion","luxury","designer","premium"]'),
(5, 'StyleHub Fit Finder - Your Perfect Size', 'AI-powered size recommendation. Never return due to wrong size again.', 'fashion', '["fashion","technology","fit","AI"]'),
(5, 'StyleHub Sustainable - Eco-Friendly Fashion', 'Shop our curated collection of sustainable and ethically-made fashion brands.', 'fashion', '["fashion","sustainable","eco","ethical"]'),
(6, 'StyleHub Accessories - Under $25', 'Amazing accessories that won''t break the bank. Jewelry, bags, hats, and more.', 'fashion', '["fashion","accessories","affordable","style"]'),
(6, 'StyleHub Shoes - BOGO 50% Off', 'Stock up on footwear. Buy one get one 50% off on all shoes, sandals, and boots.', 'fashion', '["fashion","shoes","promotion","bogo"]'),
(6, 'StyleHub Men - Business Casual Essentials', 'Upgrade your work wardrobe with our business casual collection. Free shipping over $50.', 'fashion', '["fashion","mens","business","work"]'),
(6, 'StyleHub Kids - Back to School Sale', 'Everything kids need for the new school year. Uniforms, backpacks, and shoes.', 'fashion', '["fashion","kids","school","sale"]'),
(6, 'StyleHub VIP - Early Access to Sales', 'Join StyleHub VIP for free and get 24-hour early access to all major sales events.', 'fashion', '["fashion","vip","loyalty","earlyaccess"]'),
(7, 'DragonQuest Online - Pre-order Now', 'The most anticipated RPG of the year. Pre-order for exclusive in-game items.', 'gaming', '["gaming","rpg","preorder","mmorpg"]'),
(7, 'DragonQuest Collector''s Edition Unboxed', 'See what is inside the $199 Collector''s Edition: statue, artbook, soundtrack, DLC pass.', 'gaming', '["gaming","rpg","collectors","unboxing"]'),
(7, 'DragonQuest Gameplay Trailer - 4K', 'Watch 10 minutes of exclusive DragonQuest Online gameplay in stunning 4K resolution.', 'gaming', '["gaming","rpg","trailer","gameplay"]'),
(7, 'DragonQuest - Join a Guild Today', 'Team up with players worldwide. Form guilds, raid dungeons, and conquer territories.', 'gaming', '["gaming","rpg","guild","multiplayer"]'),
(7, 'DragonQuest Free Trial - 7 Days Free', 'Try DragonQuest Online free for 7 days. No credit card required. Level cap 20.', 'gaming', '["gaming","rpg","freetrial","promotion"]'),
(8, 'BlockCrush Saga - Addictive Puzzle Game', 'The new puzzle game everyone is talking about. Over 1000 levels and counting.', 'gaming', '["gaming","mobile","puzzle","casual"]'),
(8, 'BlockCrush - VIP Pass 50% Off', 'Unlock unlimited lives, power-ups, and exclusive levels with BlockCrush VIP Pass.', 'gaming', '["gaming","mobile","vip","iap"]'),
(8, 'BlockCrush Tournament - Win Real Prizes', 'Compete in weekly BlockCrush tournaments. Top 100 players win gift cards.', 'gaming', '["gaming","mobile","tournament","prizes"]'),
(8, 'BlockCrush - Play with Friends', 'Connect with Facebook and challenge your friends. See who gets the highest score.', 'gaming', '["gaming","mobile","social","friends"]'),
(8, 'BlockCrush Merch Store Now Open', 'Official BlockCrush t-shirts, hoodies, phone cases, and plushies. Limited edition.', 'gaming', '["gaming","merchandise","store","brand"]'),
(9, 'TravelNow - Maldives from $999', 'All-inclusive Maldives vacation packages starting at $999 per person. Flights included.', 'travel', '["travel","vacation","beach","luxury"]'),
(9, 'TravelNow - European Summer Tours', 'Explore Paris, Rome, Barcelona, and more. Guided tours with local experts.', 'travel', '["travel","europe","tour","summer"]'),
(9, 'TravelNow - Last Minute Cruise Deals', 'Caribbean and Mediterranean cruises at up to 70% off. Book within 48 hours.', 'travel', '["travel","cruise","lastminute","deals"]'),
(9, 'TravelNow Business - Corporate Travel', 'Streamline business travel with TravelNow Business. Expense reporting and policy controls.', 'travel', '["travel","business","corporate","management"]'),
(9, 'TravelNow Insurance - Travel with Confidence', 'Comprehensive travel insurance from $2/day. Medical, cancellation, and baggage coverage.', 'travel', '["travel","insurance","protection","safety"]'),
(10, 'TravelNow - Weekend Getaways from $199', 'Quick escapes within driving distance. Hotels, activities, and meals bundled.', 'travel', '["travel","weekend","getaway","local"]'),
(10, 'TravelNow - Honeymoon Packages', 'Romantic honeymoon destinations. Private villas, couples spa, candlelit dinners.', 'travel', '["travel","honeymoon","romance","luxury"]'),
(10, 'TravelNow - Adventure Travel', 'Trek the Inca Trail, safari in Kenya, dive the Great Barrier Reef. Adventure awaits.', 'travel', '["travel","adventure","outdoor","extreme"]'),
(10, 'TravelNow Gift Cards - Give the Gift of Travel', 'TravelNow gift cards from $25 to $1000. The perfect gift for any occasion.', 'travel', '["travel","giftcard","gift","present"]'),
(10, 'TravelNow App - Book on the Go', 'Download the TravelNow app for exclusive mobile-only deals and instant booking.', 'travel', '["travel","app","mobile","booking"]');

INSERT INTO experiments (name, traffic_ratio, variant, hash_salt, status, description) VALUES
('GPR vs DeepFM Baseline', 0.50, 'control', 'exp-gpr-baseline-v1', 'running', 'Primary experiment comparing GPR unified architecture against traditional DeepFM scoring'),
('GPR vs LR Fallback', 0.50, 'control', 'exp-gpr-lrfallback-v1', 'running', 'Compare GPR against logistic regression fallback for timeout scenarios'),
('GPR Latency Threshold Test', 0.30, 'treatment', 'exp-gpr-latency-v1', 'running', 'Evaluate GPR performance at different latency thresholds (30ms vs 60ms)'),
('Creative Agent A/B', 0.50, 'treatment', 'exp-creative-agent-v1', 'completed', 'Test AI-generated creatives vs human-written creatives on CTR and conversion'),
('Cold Start Coverage Test', 0.20, 'treatment', 'exp-coldstart-v1', 'paused', 'Measure GPR cold start coverage on new campaigns with less than 1000 impressions');
