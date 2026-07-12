"""Consolidation adapter v1 curriculum + held-out eval.

I/O contract (new, no prior schema existed for this -- see docs/psm-model/PSM-MEMORY.md
2026-07-10 "consolidation adapter" scoping): given a NEW candidate memory and ONE existing
memory a retrieval step surfaced as plausibly related, decide whether to:
  - "store_episodic": the new memory is independent -- store it separately (existing
    vocabulary from psm_model.schema.ACTIONS; reused here to mean "not related enough to merge")
  - "update_existing": the new memory restates/elaborates/supersedes the existing one --
    merge into a single updated memory (target_memory_id set, merged_content synthesized)
  - "flag_conflict": the new memory contradicts the existing one -- flag for review, don't
    auto-merge (target_memory_id set, merged_content null)

Data sources (matching the retrieval-plan adapter's split -- same conversations, same
train/eval boundary, so this stays consistent with everything else built this session):
  - TRAIN: real update_existing/store_episodic pairs mined from conv-47/48/49/50's own
    `observation` field (the same across-session fact evolution used for retrieval-plan),
    hand-verified pair by pair -- keyword-overlap search surfaced candidates, every pair below
    was manually read and labeled, not auto-labeled at scale.
  - EVAL (conv-26, untouched by training): same mining/verification process, held out.
  - flag_conflict examples are SYNTHESIZED (genuine organic contradictions are rare in
    naturalistic LoCoMo data) using real person names from the respective train/eval
    conversations but invented contradicting content -- clearly a smaller, first-pass slice
    of this action type, flagged in the memory doc as needing expansion.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# TRAIN: update_existing (real, hand-verified pairs from conv-47/48/49/50 observations)
# ---------------------------------------------------------------------------
_TRAIN_UPDATE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("jolene-pet-susie",
     "Jolene has a pet named Susie who has been with her for two years and brings her comfort and peace.",
     "Jolene adopted Susie two years ago when feeling lonely."),
    ("sam-health-issues",
     "Sam is dealing with health issues that have been rough on him.",
     "Sam has been stressed with phone issues on top of the health scare."),
    ("jolene-yoga-stress",
     "Jolene practices yoga and meditation to relax and stay focused.",
     "Yoga and meditation have helped Jolene with stress and keeping centered."),
    ("jolene-project-done",
     "Jolene recently completed a tough engineering project.",
     "Jolene is working on a big project which is tough but exciting to watch take shape."),
    ("evan-family-trip",
     "Evan took his family on a road trip to Jasper last weekend, driving through the Icefields Parkway.",
     "Evan went on a trip to the Rockies with his family."),
    ("deborah-yoga-guide",
     "Deborah made a meditation guide for her yoga retreat.",
     "Deborah did yoga and meditation to relax last Friday."),
    ("calvin-studio",
     "Calvin has a studio setup where he works, surrounded by music videos, concerts, and documentaries for inspiration.",
     "Calvin wrote some new tunes and had studio sessions last week, excited to collaborate and share the music."),
    ("dave-cars-origin",
     "Dave started working on cars at the age of ten after finding an old car in a neighbor's garage.",
     "Working on cars is like therapy for Dave and a way to get away from everyday stress."),
    ("calvin-music-passion",
     "Calvin is passionate about music and performing, finding it like his purpose and passion.",
     "Calvin gifted a necklace with a diamond pendant as a reminder of his passion for music."),
    ("deborah-moms-house",
     "Deborah visited her mom's house last month, which holds a special place in her heart as a symbol of her mom's strength and love.",
     "Deborah's mother had a special bench near the window in her house where she used to sit every morning to take in the view."),
    ("jolene-exams-stress",
     "Jolene finds it difficult to manage time and stay organized during exams and deadlines.",
     "Jolene feels overwhelmed by exams and deadlines."),
    ("deborah-encourages-jolene",
     "Deborah encourages Jolene to focus on her goals, not give up, and offers help in starting mindfulness practice.",
     "Deborah reminds Jolene that efforts will bear fruit and encourages her not to give up."),
    ("calvin-collab-excited",
     "Calvin is working on music collaborations with Japanese artists and is excited about it.",
     "Calvin wrote some new tunes and had studio sessions last week, excited to collaborate and share the music."),
    # Second batch (2026-07-10, rebalancing after v1 collapsed to always-predict update_existing)
    ("dave-supportive-calvin-tour",
     "Dave is supportive and encouraging of Calvin's music career.",
     "Dave enjoys the music scene in Boston and is looking forward to Calvin's tour and performance in the city."),
    ("dave-supportive-friends",
     "Dave acknowledges the importance of having supportive friends since the start to help artists like Calvin.",
     "Dave is supportive and encouraging of Calvin's music career."),
    # Fourth batch (2026-07-11, round 4 expansion -- fresh John/James material from conv-47)
    ("james-ned-adoption",
     "James adopted a pup from a shelter in Stamford last week and named it Ned, making his days happier.",
     "James has a dog named Ned that he adopted and can't imagine life without."),
    ("james-pets-introduced",
     "James introduced three pets - Max, Daisy, and a new pup named Ned, who are slowly adapting and bonding together.",
     "James took a great photo of his pets bonding together."),
    ("john-nonprofit-motivation",
     "John volunteered his programming skills for a social cause, creating a software tool for a charitable foundation to streamline their operations.",
     "This experience gave John a clearer sense of purpose and motivated him to potentially pursue a career in the non-profit sector."),
    ("james-chess-advice",
     "James has previously played chess and acknowledges its strategic nature.",
     "James shared advice with John on improving in chess by studying opening moves and analyzing games."),
    ("john-siblings-coding",
     "John helps his younger siblings with programming and is proud of their progress",
     "John has been teaching his siblings coding, and they are creating their own programs."),
    ("jolene-games-passion-origin",
     "Jolene's passion for video games started when she was 10, which her parents supported.",
     "Jolene enjoys playing video games and particularly likes the game \"Detroit\" on the console."),
    ("calvin-music-selfdiscovery",
     "Calvin expresses himself through music, viewing it as a form of therapy and self-expression.",
     "Calvin finds experimenting with different music genres an exciting process of self-discovery and growth."),
    ("jolene-room-meaning",
     "Jolene's room is her haven for peace and rest where she goes to relax and recharge after a busy day.",
     "Jolene has a room in her mother's house where she has many memories."),
    ("dave-fixing-things",
     "Dave finds fulfillment in fixing things and enjoys the feeling of making something whole again.",
     "Restoring things can be tough for Dave, but the feeling of accomplishment it gives him is great."),
    # Third batch (2026-07-10, round 3 expansion)
    ("calvin-appreciates-dave-cars",
     "Calvin appreciates Dave's talent in fixing cars and finds it inspiring.",
     "Calvin appreciates Dave's hard work and dedication in achieving his dream of opening a car maintenance shop."),
    ("deborah-yoga-peaceful-practice",
     "Deborah finds doing yoga on the beach with the ocean, sand, and fresh air peaceful and a perfect way to take care of herself.",
     "Deborah finds instrumental tracks with mellow melodies and rhythms helpful for creating a peaceful vibe during her practice."),
    ("dave-classic-cars-dream",
     "Dave's dream is to work on classic cars due to his love for their design and engineering.",
     "Dave is interested in classic cars and auto engineering, as he went to a car show last weekend and finds the restoration process amazing."),
    ("john-nonprofit-considering",
     "John is considering going into non-profit work and using his skills and passions for causes he cares about.",
     "John is considering volunteer roles and potentially a career in the non-profit sector."),
    ("jolene-surfing-progression",
     "Jolene just started learning about surfing but hasn't gone yet.",
     "Jolene is planning on learning to surf, has been gathering information, and got a beginners' guide to surfing."),
    ("calvin-frankocean-tour-connecting",
     "Calvin is on tour with Frank Ocean and finds it incredible, connecting with the crowd.",
     "Calvin plans to visit the snowy peaks in Japan after his tour with Frank Ocean ends."),
    ("calvin-dave-dreams-support",
     "Calvin is determined to make his dreams come true and appreciates Dave's support and encouragement.",
     "Calvin acknowledges Dave's guts and ambition and supports him in pursuing his dreams."),
    ("dave-photography-hobby",
     "Dave has taken up photography as a new hobby and enjoys capturing the scenery around him.",
     "Dave has been getting into photography recently and has taken some great shots."),
    ("jolene-engineering-relationship",
     "Jolene is finding it challenging to juggle her engineering studies, relationship, and personal growth.",
     "Jolene and her partner met in an engineering class in college and their romantic relationship grew from a friendship."),
    ("john-james-game-mentorship",
     "John advises James about the game Cyberpunk 2077, highlighting the significance of making the right choices in the game.",
     "John encourages James to continue with game development and expresses belief in his talents."),
    # Fifth batch (2026-07-12, round 5 expansion -- fresh material from conv-30/conv-41, two
    # LoCoMo conversations untouched by any prior consolidation or retrieval-plan training/eval;
    # mined specifically to grow update_existing beyond the 2x store_episodic boost that was
    # masking it, per the lesson that a full-retrain-at-high-LR on top of a small/imbalanced
    # curriculum (v6) collapsed toward the majority class instead of fixing the boundary)
    ("john-dog-max-memory",
     "John cherishes memories of his pet Max, including a camping trip where they hiked, swam, and made great memories.",
     "John recently had to say goodbye to his dog, Max, who was a part of their family for 10 years."),
    ("maria-volunteering-origin",
     "Maria volunteers at a homeless shelter, which she started about a year ago after witnessing a struggling family on the streets.",
     "Maria volunteers at a homeless shelter and recently started aerial yoga."),
    ("john-education-infrastructure-passion",
     "John has been thinking about how education and infrastructure shape communities.",
     "John's passion in politics revolves around improving education and infrastructure in the community."),
    ("john-firefighting-brigade",
     "John is now part of the fire-fighting brigade and is enthusiastic about helping the community.",
     "John joined a firefighting brigade to give back to his community."),
    ("john-veteran-support-project",
     "John is part of a virtual support group advocating for the military and has involved family and friends in supporting veterans.",
     "John is currently working on a project to support military veterans and is trying to get a petition going."),
    ("maria-spreading-positivity",
     "Maria plans to keep spreading positivity.",
     "Maria believes that spreading kindness and positivity is her way of impacting the world."),
    ("john-yoga-class",
     "John started a weekend yoga class with a colleague and finds it awesome for his mental and physical wellbeing.",
     "John's colleague invited him to a beginner's yoga class."),
    ("jon-dance-studio-running",
     "Jon is a dancer who runs his own dance studio.",
     "Jon is starting his own dance studio due to his passion for dancing."),
    ("jon-dance-studio-search",
     "Jon is searching for a dance studio location and is determined to find the right spot.",
     "Jon is looking for the ideal spot for his dance studio and is considering features like size, natural light, and flooring."),
    ("gina-clothing-store",
     "Gina owns a store where she sells fashion pieces.",
     "Gina started her own clothing store and finds taking risks scary but rewarding."),
)

# ---------------------------------------------------------------------------
# TRAIN: store_episodic (related topic, but genuinely distinct facts -- must NOT merge)
# ---------------------------------------------------------------------------
_TRAIN_STORE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("deborah-art-vs-retreat",
     "Deborah attended a yoga retreat near her mom's place last week and found it life-changing.",
     "Deborah attended an art show with a friend which she found cool and inspiring."),
    ("calvin-car-vs-advice",
     "Calvin received advice from a music producer to stay true to himself and sound unique, which he found motivating.",
     "Calvin recently got a new car which is a luxury car and a dream come true for him."),
    ("dave-boston-vs-carshop",
     "Dave is excited about the journey of running his car shop and is looking forward to what the future holds.",
     "Dave is looking forward to showing Calvin his favorite spots in Boston, especially in terms of food and music."),
    ("calvin-frankocean-vs-dave",
     "Calvin is excited about jamming music together with Dave.",
     "Calvin met Frank Ocean at a music festival in Tokyo where they clicked and recorded a song together."),
    ("jolene-games-vs-partner",
     "Jolene's partner enjoys playing the console she bought.",
     "Jolene spends time playing video games with her partner to relax after a long day."),
    ("calvin-mansion-vs-inspired",
     "Listening to struggles people go through inspires Calvin in his music, leading him to dig deeper into capturing feelings.",
     "Calvin is working on transforming a Japanese mansion into a recording studio, which is his dream for creating music with other artists."),
    ("dave-appreciates-two-things",
     "Dave appreciates when Calvin notices the effort he puts into his work.",
     "Dave appreciates Calvin's music and is excited for his tour."),
    ("calvin-boston-vs-tokyo",
     "Calvin is excited for an upcoming performance in Tokyo this month to showcase his music to a new crowd and expand his following.",
     "Calvin is planning an upcoming trip to Boston after finishing the Frank Ocean tour to explore the music scene there."),
    # Second batch (2026-07-10, rebalancing)
    ("deborah-house-vs-event",
     "Deborah went to a cool event last week aimed at supporting each other.",
     "Deborah visited her mother's old house last week which holds special memories as her mother passed away a few years ago."),
    ("deborah-teaching-vs-retreat",
     "Deborah is preparing for a yoga retreat with friends to find peace and understanding.",
     "Deborah finds teaching yoga calming and derives happiness from giving people peace and awareness."),
    ("deborah-bali-vs-routine",
     "Deborah shared her favorite gentle yoga flow routine focused on breathing and grounding to find chill.",
     "Deborah traveled to Bali last year, one of her favorite places, for peace and yoga."),
    ("calvin-mansion-vs-newsounds",
     "Calvin enjoys trying out new sounds and pushing boundaries in his music studio to stay ahead.",
     "Calvin is working on transforming a Japanese mansion into a recording studio, which is his dream for creating music with other artists."),
    ("deborah-goal-vs-oldhome",
     "Deborah visits her old home, where her mom passed away, to find peace and feel her mother's presence.",
     "Deborah's goal is to keep teaching yoga and supporting her community to help people find peace and joy."),
    ("deborah-house-vs-beach",
     "The beach where Deborah got married and discovered her love for surfing holds special memories of joy and peace for her.",
     "Deborah visited her mother's old house last week which holds special memories as her mother passed away a few years ago."),
    ("dave-encourages-vs-frankocean",
     "Dave commented on the opportunity Calvin had to meet Frank Ocean at a music festival in Tokyo.",
     "Dave encourages Calvin in his music goals, supports his dreams, and provides words of motivation and belief."),
    ("john-project-vs-boardgame",
     "John worked with a game developer on a project to create an online board game over the weekend.",
     "John worked on a project with someone from the online group last week."),
    ("jolene-music-vs-pet-comfort",
     "Jolene finds comfort and distraction through her pet Susie and video games during tough times.",
     "Jolene finds music helpful during her yoga practice."),
    # Third batch (2026-07-10, round 3 expansion)
    ("deborah-scents-vs-recommend",
     "Deborah recommended self-care routines like yoga to Jolene to stay balanced and grounded.",
     "Deborah enjoys scents like lavender and rosemary during her yoga practice."),
    ("james-pythonc-vs-strategy",
     "James is interested in creating a strategy game similar to Civilization.",
     "James has worked with Python and C++, building websites and creating game mods."),
    ("calvin-neverjapan-vs-neverboston",
     "Calvin has never been to Boston before but is excited to visit the amazing parks next month.",
     "Calvin has never been to Japan before but is fascinated by the traditions and culture."),
    ("dave-concert-vs-festival",
     "Dave recently attended a music festival and enjoyed the energy, music, and crowd.",
     "Dave attended a rock concert in Boston recently and enjoyed the atmosphere."),
    ("calvin-ferrari-vs-tokyo",
     "Calvin is excited to explore Shibuya Crossing and Shinjuku in Tokyo and is looking forward to trying the amazing food there.",
     "Calvin recently acquired a new Ferrari and is looking forward to thrilling rides and journeys."),
    ("deborah-familyphotos-vs-pets",
     "Deborah mentioned pets as a source of love and comfort during tough times.",
     "Deborah finds peace in looking at family photos during difficult times."),
    ("calvin-frankocean-snowypeaks-vs-chemistry",
     "Calvin and Frank Ocean have a great chemistry on stage, and Calvin feels fortunate about it.",
     "Calvin plans to visit the snowy peaks in Japan after his tour with Frank Ocean ends."),
    ("dave-tokyo-vs-frankocean",
     "Dave expressed excitement about Calvin continuing collaboration with Frank Ocean.",
     "Dave expressed interest in taking a trip to Tokyo after seeing Calvin's picture of the night skyline."),
    ("calvin-showcar-vs-musicjam",
     "Calvin is looking forward to creating something special during the music jam session with Dave.",
     "Calvin has a car that he has put a lot of work into and is looking forward to showing it to Dave when he visits Boston."),
    ("evan-painting-vs-helpingsam",
     "Evan is willing to help Sam get started with painting by recommending supplies and planning a painting session.",
     "Evan recently started taking painting classes and enjoys expressing himself through art."),
    ("james-extremesports-vs-gaminggenre",
     "James is interested in trying out a new gaming genre and mentioned considering the sports genre.",
     "James is interested in extreme sports, recently trying rope jumping from a height of 150 meters and surfing."),
    ("james-sketchgame-vs-footballsim",
     "James is currently working on a football simulator project, specifically on collecting player databases.",
     "James is working on a project to turn his childhood sketches of a main character into a computer game, combining his passions for gaming and storytelling."),
    ("evan-believes-support-vs-loveatfirstsight",
     "Evan believes in love at first sight and felt a spark when he met his partner.",
     "Evan believes it makes a big difference to have support when trying new things."),
    ("james-believes-football-vs-dogs",
     "James already has three dogs at home and believes having more than three dogs is too much.",
     "James believes there is no sport better than football and no club better than Liverpool."),
    ("john-souvenir-vs-skateboarding",
     "John has a picture from elementary school with James related to skateboarding, showing they were friends who enjoyed skateboarding together.",
     "John will be waiting for James to return from his trip and expressed excitement about a souvenir."),
    # Fourth batch (2026-07-11, round 4 expansion -- fresh John/James material from conv-47
    # not touched by the earlier batches above)
    ("john-metaldetector-vs-drums",
     "John recently got into a new hobby of using a metal detector on the beach to look for items.",
     "John plays drums and has been playing for only a month."),
    ("james-tournament-vs-instrument",
     "James joined an online gaming tournament and made it to the semifinals, winning some rounds.",
     "James is learning to play a musical instrument and has been at it daily, seeing improvements."),
    ("john-japan-vs-newfriends",
     "John visited Japan last, where he was impressed by the technologically advanced megacities and delicious street food.",
     "John met three new friends in his programming course last Tuesday and is excited to expand his social circle."),
    ("james-virtualworld-vs-travel",
     "James worked on a programming project combining gaming with programming, creating a virtual world inspired by Witcher 3 with a game character he designed.",
     "James has visited Italy, Turkey, and Mexico besides his permanent residence and found Italy to be very beautiful with delicious food."),
    ("john-freelance-vs-pizza",
     "John is currently taking on freelance programming to hone his coding skills.",
     "John loves Hawaiian pizza for its sweet and salty combination."),
    ("james-cookingclass-vs-samantha",
     "James signed up for a cooking class two days ago to learn something new.",
     "James asked Samantha to be his girlfriend at the theater, and she agreed."),
    ("john-boardgames-vs-drums",
     "John recently got into board games and found them to be a lot of fun.",
     "John used to play drums when he was younger."),
    ("james-dream-vs-streaming",
     "James had a creative dream a few weeks ago that led to interesting thoughts.",
     "James has started streaming games and hopes everything works out."),
    ("john-startup-vs-chess",
     "John started a new startup focusing on portable smokers and has already welded one from metal.",
     "John recently started playing chess to improve his strategic thinking."),
    ("james-roadtrip-vs-gaminggear",
     "James started a road trip with his family and dogs yesterday, enjoying exploring new places and nature.",
     "James got a cool video card last week and is excited to use it for playing RPGs."),
)

# ---------------------------------------------------------------------------
# EVAL (conv-26, never used in training): update_existing
# ---------------------------------------------------------------------------
_EVAL_UPDATE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("melanie-kids-park",
     "Melanie took her kids to a park and enjoyed seeing them have fun exploring and playing.",
     "Melanie loves spending time with her kids and seeing the joy in their eyes when exploring new things."),
    ("caroline-adoption-dream",
     "Caroline sees adoption as a way to share her love and provide a safe, loving home for kids in need.",
     "Caroline is researching adoption agencies with the dream of having a family and providing a loving home to kids in need."),
    ("caroline-career-counseling",
     "Caroline is considering a career in counseling and mental health to help others.",
     "Caroline is planning to continue her education and explore career options in counseling or mental health to support those with similar issues."),
    ("caroline-adoption-progress",
     "Caroline passed the adoption agency interviews last Friday and is excited about building her own family through adoption.",
     "Caroline attended a council meeting for adoption last Friday and found it inspiring and emotional."),
    ("melanie-pottery-plate",
     "Melanie made a plate in pottery class and finds pottery relaxing and creative.",
     "Melanie made a black and white bowl in her pottery class which she is proud of."),
    ("caroline-counseling-passionate",
     "Caroline has been looking into counseling or mental health work and is passionate about helping people and making a positive impact.",
     "Caroline is considering a career in counseling and mental health to help others."),
    ("melanie-pottery-calming",
     "Melanie uses painting and pottery as a calming and satisfying creative outlet.",
     "Melanie is a big fan of pottery and finds it calming and creative."),
    ("caroline-safe-home-vision",
     "Caroline's vision for the future includes creating a safe and loving home for needy kids to experience love and acceptance.",
     "Caroline sees adoption as a way to share her love and provide a safe, loving home for kids in need."),
    # Fourth batch (2026-07-11, round 4 eval expansion -- fresh conv-26 material)
    ("melanie-pottery-elaboration",
     "Melanie signed up for a pottery class and finds it therapeutic for self-expression and creativity.",
     "Melanie is a big fan of pottery and finds it calming and creative."),
    ("caroline-counseling-motivation",
     "Caroline is considering a career in counseling and mental health, particularly working with trans people to help them accept themselves and support their mental health.",
     "Caroline's motivation to pursue counseling comes from her own journey, the support she received, and the positive impact counseling had on her life."),
)

# ---------------------------------------------------------------------------
# EVAL (conv-26): store_episodic
# ---------------------------------------------------------------------------
_EVAL_STORE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("caroline-pride-events",
     "Caroline had a great time with the whole gang at the Pride fest last year and values supportive friends.",
     "Caroline and her mentee had a great time at the LGBT pride event the previous month."),
    ("melanie-painting-vs-reading",
     "Melanie continued expressing herself through reading and painting during her break from pottery.",
     "Melanie expresses herself through painting and values art for showing who we really are and getting in touch with ourselves."),
    ("caroline-supportgroup-vs-youthcenter",
     "Caroline had the opportunity to volunteer at an LGBTQ+ youth center and found it gratifying to support and guide the young people there.",
     "Caroline attended an LGBTQ support group recently and found the transgender stories inspiring."),
    ("caroline-book-vs-mentalhealth",
     "Caroline mentioned the importance of finding peace and mental health through expressions of authentic self.",
     "According to 'Becoming Nicole,' Caroline learned the importance of self-acceptance, finding support, and the existence of hope and love."),
    ("caroline-friends-vs-help",
     "Caroline received invaluable help from friends, family, and role models during the process of finding acceptance.",
     "Caroline has known her friends for 4 years, since moving from her home country, and values their love and help, especially after a tough breakup."),
    ("caroline-conference-vs-parade",
     "Caroline attended a pride parade recently and felt inspired by the community's energy and support for LGBTQ rights.",
     "Caroline attended an LGBTQ conference recently and felt accepted and supported, emphasizing the importance of fighting for trans rights and spreading awareness."),
    ("melanie-camping-vs-meteor",
     "Melanie and her family watched the Perseid meteor shower during a camping trip last year and it was a memorable experience.",
     "Melanie went camping with her family in the mountains last week and had a great time exploring nature, roasting marshmallows, and hiking."),
    # Fourth batch (2026-07-11, round 4 eval expansion -- fresh conv-26 material)
    ("caroline-pridewalk-vs-piano",
     "Caroline attended an LGBTQ+ pride parade last week and felt a sense of belonging and happiness.",
     "Caroline is currently learning the piano to get creative."),
    ("melanie-pottery-vs-camping",
     "Melanie signed up for a pottery class and finds it therapeutic for self-expression and creativity.",
     "Melanie and her family enjoy camping at the beach as it brings them closer together."),
)

# ---------------------------------------------------------------------------
# Synthesized flag_conflict examples (real names, invented contradicting content;
# train names from conv-47/48/49/50, eval names from conv-26 -- content itself is
# synthetic, not real conv-26 conversation text, so no eval contamination)
# ---------------------------------------------------------------------------
_TRAIN_CONFLICT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("john-job-conflict",
     "John mentioned he just started a new job as a backend engineer at a fintech startup.",
     "John said his job is a marketing coordinator at a retail company."),
    ("james-city-conflict",
     "James said he's been living in Denver for the past two years.",
     "James mentioned he currently lives in Portland."),
    ("dave-allergy-conflict",
     "Dave mentioned he's allergic to peanuts and avoids them carefully.",
     "Dave said he loves peanut butter sandwiches and eats them daily."),
    ("jolene-pet-conflict",
     "Jolene said she doesn't have any pets right now.",
     "Jolene has a pet named Susie who has been with her for two years."),
    ("calvin-marital-conflict",
     "Calvin mentioned he's been married for five years.",
     "Calvin said he's never been married and prefers being single."),
    # Second batch (2026-07-10, rebalancing)
    ("deborah-diet-conflict",
     "Deborah mentioned she's been vegan for three years and avoids all animal products.",
     "Deborah said she had a steak dinner last night and loves eating meat."),
    ("sam-kids-conflict",
     "Sam mentioned he doesn't have any children.",
     "Sam said his two kids just started school this year."),
    ("evan-job-conflict",
     "Evan mentioned he was recently laid off and is job hunting.",
     "Evan said he's been at the same company for ten years and loves his job security."),
    ("jolene-relationship-conflict",
     "Jolene said she's currently single and not dating anyone.",
     "Jolene mentioned her partner surprised her with tickets to a concert."),
    ("dave-location-conflict",
     "Dave said he's lived in the same house in Boston his whole life.",
     "Dave mentioned he just moved to a new apartment across town last month."),
    # Third batch (2026-07-10, round 3 expansion)
    ("calvin-siblings-conflict",
     "Calvin mentioned he's an only child with no siblings.",
     "Calvin said his older sister just visited him in Tokyo."),
    ("james-diet-conflict",
     "James mentioned he's a strict vegetarian and hasn't eaten meat in years.",
     "James said he grilled steaks for the whole group last weekend."),
    ("john-education-conflict",
     "John said he never finished college and started working right after high school.",
     "John mentioned he just graduated with a master's degree in computer science."),
    ("jolene-car-conflict",
     "Jolene said she doesn't own a car and relies on public transit.",
     "Jolene mentioned she just got her car back from the shop after an oil change."),
    ("dave-sports-conflict",
     "Dave said he's never been interested in sports at all.",
     "Dave mentioned he's been training for a marathon for the past six months."),
)
_EVAL_CONFLICT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("caroline-city-conflict",
     "Caroline mentioned she moved to Austin last year.",
     "Caroline said she still lives in her hometown and has never moved."),
    ("melanie-job-conflict",
     "Melanie said she works as a full-time nurse at a hospital.",
     "Melanie mentioned she is a stay-at-home parent and hasn't worked outside the home in years."),
    ("caroline-pet-conflict",
     "Caroline mentioned she has never owned a pet and isn't interested in having one.",
     "Caroline said her dog just had puppies and she's finding homes for them."),
    ("melanie-diet-conflict",
     "Melanie said she's been a strict vegetarian since childhood.",
     "Melanie mentioned she grilled burgers for the family barbecue last weekend."),
)


def _memory(content: str) -> dict[str, Any]:
    return {"content": content, "type": "episodic"}


def _row(row_id: str, new_text: str, existing_text: str, *, action: str) -> dict[str, Any]:
    existing_id = f"mem-{row_id}-existing"
    if action == "update_existing":
        expected = {
            "action": "update_existing",
            "target_memory_id": existing_id,
            "merged_content": f"{existing_text} {new_text}",
            "reasoning": "New memory restates/elaborates the same underlying fact as the existing one; merge into a single updated memory.",
        }
    elif action == "flag_conflict":
        expected = {
            "action": "flag_conflict",
            "target_memory_id": existing_id,
            "merged_content": None,
            "reasoning": "New memory contradicts the existing one on the same subject; flag for review instead of auto-merging.",
        }
    else:
        expected = {
            "action": "store_episodic",
            "target_memory_id": None,
            "merged_content": None,
            "reasoning": "New memory is a distinct fact, not a restatement or contradiction of the existing one; store independently.",
        }
    return {
        "id": f"consolidation-{row_id}",
        "task": "consolidate",
        "input": {
            "operation": "consolidate",
            "new_memory": _memory(new_text),
            "existing_memory": {"id": existing_id, **_memory(existing_text)},
        },
        "expected": expected,
        "source": f"consolidation_locomo:{action}",
    }


def build_consolidation_train_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, new_text, existing_text in _TRAIN_UPDATE_PAIRS:
        rows.append(_row(row_id, new_text, existing_text, action="update_existing"))
    for row_id, new_text, existing_text in _TRAIN_STORE_PAIRS:
        rows.append(_row(row_id, new_text, existing_text, action="store_episodic"))
    for row_id, new_text, existing_text in _TRAIN_CONFLICT_PAIRS:
        rows.append(_row(row_id, new_text, existing_text, action="flag_conflict"))
    return rows


def build_consolidation_eval_cases() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for row_id, new_text, existing_text in _EVAL_UPDATE_PAIRS:
        row = _row(row_id, new_text, existing_text, action="update_existing")
        cases.append(_case_from_row(row))
    for row_id, new_text, existing_text in _EVAL_STORE_PAIRS:
        row = _row(row_id, new_text, existing_text, action="store_episodic")
        cases.append(_case_from_row(row))
    for row_id, new_text, existing_text in _EVAL_CONFLICT_PAIRS:
        row = _row(row_id, new_text, existing_text, action="flag_conflict")
        cases.append(_case_from_row(row))
    return {"cases": cases}


def _case_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "suite": "consolidation_locomo_holdout",
        "newMemory": row["input"]["new_memory"],
        "existingMemory": row["input"]["existing_memory"],
        "expected": row["expected"],
    }
