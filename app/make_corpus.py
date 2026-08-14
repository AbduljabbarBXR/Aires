"""Generate the child's first real curriculum corpus:
2000 high-frequency words + ~8000 relationship sentences.

The sentences are the RELATIONSHIP data: they wire words to words
(dog-bone, apple-red) and words to letters, through actual usage.
"""

import json
import random
from pathlib import Path

from config import VAULT_DIR

rng = random.Random(42)

CUR = VAULT_DIR / "curriculum"
CUR.mkdir(parents=True, exist_ok=True)

# ---- the 2000-word vocabulary, by category (the columns) ----
WORDS = {
    "people": "mother father sister brother baby boy girl man woman friend family teacher doctor nurse farmer driver child kid parent grandpa grandma uncle aunt cousin wife husband person name".split(),
    "body": "head face eye ear nose mouth tooth teeth tongue hand foot arm leg knee finger hair neck shoulder back belly chest skin blood heart".split(),
    "animals": "dog cat bird fish cow horse pig sheep chicken duck goat rabbit mouse rat lion tiger bear elephant monkey giraffe zebra wolf fox deer snake frog bee ant spider butterfly owl eagle hen rooster turkey whale dolphin shark crab turtle".split(),
    "food": "apple banana bread egg milk rice meat soup cake cheese water juice orange grape lemon peach pear cherry strawberry melon potato tomato onion carrot corn bean pea nut honey sugar salt pepper butter oil tea coffee".split(),
    "home": "house room door window wall floor roof bed table chair sofa lamp clock mirror towel soap cup plate bowl spoon fork knife glass bottle bag box basket shelf drawer curtain rug carpet pillow blanket sheet".split(),
    "objects": "ball book pen pencil paper bag box car bus train plane bike boat truck phone computer keyboard mouse screen lamp key lock bell button wire plug battery tool hammer nail screw rope string thread needle cloth".split(),
    "nature": "sun moon star sky cloud rain snow wind storm thunder lightning rainbow river lake sea ocean mountain hill tree flower grass leaf branch root stone rock sand mud dust fire smoke air earth world island desert forest garden field farm".split(),
    "weather": "hot cold warm cool sunny cloudy rainy windy snowy foggy dry wet bright dark".split(),
    "time": "day night morning afternoon evening week month year hour minute second today tomorrow yesterday now later early late spring summer autumn winter birthday holiday weekend".split(),
    "numbers": "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand first second third fourth fifth last".split(),
    "colors": "red blue green yellow black white brown pink orange purple gray grey gold silver".split(),
    "clothes": "shirt dress pants shorts skirt coat jacket sweater hat cap shoe boot sock glove scarf belt pocket button collar".split(),
    "school": "school class lesson teacher student book page word letter number question answer test exam homework study learn teach read write listen speak talk say ask think know remember forget".split(),
    "actions": "go come run walk jump sit stand lie fall turn open close push pull give take bring carry put get make build break cut wash clean cook eat drink sleep wake play work help love like want need see look watch hear touch smell taste feel".split(),
    "feelings": "happy sad angry afraid scared tired hungry thirsty sick well fine good bad nice kind funny strong weak big small tall short long fast slow new old young easy hard loud quiet".split(),
    "places": "city town village street road park shop market store bank church hospital airport station school house farm zoo museum cinema restaurant hotel beach bridge".split(),
    "travel": "map ticket luggage passport train bus plane ship taxi ride travel visit arrive leave return trip journey".split(),
    "family": "home family love hug kiss laugh cry smile laugh gift party music song dance game toy friend together".split(),
    "technology": "computer phone internet email message photo video music game app screen button camera video call".split(),
    "grammar": "i you he she it we they me him her us them my your his her its our their this that these those a an the and but or so because if when where why how what who which not no yes".split(),
    "prepositions": "in on at to from with without for by about against between through during before after above below over under again further then once here there all any both each few more most other some such only own same so than too very just also".split(),
}

# ---- extended vocabulary (to reach ~2000 words) ----
EXTRA = {
    "animals2": "horse pony donkey goat pig cow bull calf hen rooster goose duck turkey crow sparrow robin pigeon parrot peacock penguin ostrich flamingo pelican seagull eagle hawk vulture swan heron stork crane bat owl skunk raccoon squirrel beaver otter badger hedgehog mole porcupine kangaroo koala wombat platypus dolphin whale shark octopus squid jellyfish starfish seahorse crab lobster shrimp snail slug worm caterpillar cocoon beetle ladybug grasshopper cricket praying mantis dragonfly damselfly wasp hornet mosquito fly moth firefly centipede millipede scorpion tarantula gecko iguana chameleon crocodile alligator lizard newt salamander toad frog tadpole".split(),
    "food2": "carrot potato tomato onion garlic ginger pepper chili cucumber pumpkin squash zucchini broccoli cauliflower cabbage spinach lettuce kale celery radish turnip beet corn rice pasta noodle cereal oatmeal pancake waffle muffin cookie biscuit cracker chip popcorn pretzel candy chocolate cake pie pastry dough bread toast bagel croissant cheese yogurt cream butter margarine jam jelly honey syrup sauce gravy dressing marinade pickle olives avocado mango pineapple coconut papaya pomegranate fig date raisin almond walnut peanut cashew pistachio sesame sunflower seed flour sugar salt pepper cinnamon vanilla cocoa coffee tea juice smoothie milkshake soda lemonade".split(),
    "objects2": "spoon fork knife chopsticks plate bowl cup mug glass saucer pot pan kettle toaster blender mixer grater peeler bottle jar tin can package box carton basket tray napkin towel sponge scrubber broom mop dustpan bucket hose sprinkler rake shovel trowel saw hammer nail screw bolt nut wrench pliers screwdriver drill chisel plane sandpaper ladder step stool paintbrush roller tape glue scissors stapler paperclip rubber band envelope stamp folder binder notebook notepad sticky note calendar ruler compass protractor eraser sharpener pencil crayon marker highlighter chalk whiteboard".split(),
    "nature2": "valley canyon cliff plateau plain prairie savanna desert dune oasis swamp bog marsh lagoon inlet bay gulf strait peninsula island archipelago coastline shore beach harbor pier dock lighthouse reef atoll tide wave current foam spray mist fog haze dew frost ice glacier iceberg volcano eruption magma lava ash crater geyser spring well waterfall cascade rapids brook creek stream pond lake reservoir dam canal aqueduct bridge tunnel road path trail track railway highway freeway avenue boulevard lane alley courtyard plaza square market fair carnival festival parade".split(),
    "verbs2": "accept achieve act add admit agree allow announce apologize appear apply argue arrange arrive ask attack attend avoid bake balance bang base bathe battle beat beg behave believe belong bend bet bite blame bless blow board boast boil borrow bounce bow brake break breathe breed bring broadcast brush build burn bury buy calculate call camp cancel capture care carry carve cash catch celebrate change chase check cheer choose chop claim clap clean clear climb close collect comb come command compare compete complain complete concentrate confess confuse connect consider contain continue cook copy correct cost count cover crack crash crawl create creep cross crush cry cure curl curve cut cycle dance dare deal decay deceive decide decorate delay deliver demand deny depend describe deserve design destroy detect develop die dig dip direct disappear discover discuss dive divide do donate double doubt drag drain draw dream dress drill drink drip drive drop dry dump earn eat echo edit educate elect email embarrass employ empty encourage end enjoy enter entertain escape examine exchange excuse exercise exist expand expect explain explode explore export express extend face fade fail fall fancy fasten favor fear feed feel fetch fight fill film find finish fire fit fix flap flash float flood flow fly fold follow fool forbid force forget forgive form found frame free freeze fry gather gaze generate gift give glance glow glue go grab greet grin grind grip grow guarantee guard guess guide handle hang happen harm hate haul heal hear heat help hide hike hint hire hit hold hook hop hope hunt hurry hurt imagine impress improve include increase influence inform inject injure insist instruct intend interest interrupt introduce invent invite iron irritate jog join joke judge juggle jump keep kick kid kiss kneel knit knock knot know label land last laugh launch lay lead leak lean leap learn leave lend let lick lie lift light like limit link list listen live load lock lodge look loosen lose love lower luck lump lunge lure march mark marry mash match mate matter mean measure meet melt memorize mend mention merge might milk mind mine miss mix moan model move mow multiply murder name nap narrate need nest nod note notice nurse obey observe obtain occupy occur offer oil open operate order organize own pack paddle paint park part pass paste pat pause pay peek peel peep perform permit pick picture pile pin pinch pine pipe pitch place plan plant play please plug point poke polish pop possess post pour practice praise pray preach prefer prepare present press pretend prevent print process produce promise promote propose protect protest prove provide pull punish push qualify question queue quit race rage raid rain raise range rank reach react read realize reason recall receive recognize record recover recycle reduce refer reflect refuse regret reject relax release rely remain remember remind remove rent repair repeat replace reply report represent request rescue reserve resist respect rest restore retire return reveal review reward ride ring rinse rise risk rob rock roll root rotate rough round row rub ruin rule run rush sail satisfy save scare scatter schedule scold score scratch scream scrub search seat secure see seek seem select sell send serve settle sew shade shake share shave shed shine ship shiver shock shoot shop shout show shower shrink shut sigh sign signal sing sink sip sit size skate ski skip slam slap sleep slide slip smell smile smoke snap sneeze sniff snore soar sob solve soothe sort sound sow space spare spark speak speed spell spend spill spin spit splash split spoil sponge spot spray spread spring sprinkle sprint squeeze stack staff stamp stand stare start starve state stay steady steal steam steer step stick sting stir stock stop store storm story straight strain stray stretch strike string strip stroke stroll struggle study stuff stumble stump stunt subject submit subtract succeed suck suffer suggest suit summarize supply support suppose surf survive suspect swallow swear sweat sweep swell swim swing switch sympathize tag take talk tap taste teach tear tease tell tempt tend test thank thaw think throw tick tickle tidy tie time tip tire toast tolerate tone toss touch tour tow trace track trade train transfer trap travel treasure treat tremble trick trim trip trot trouble trust try tuck tune turn twist type underline undo unite unlock unload unpack untie update use vanish visit volunteer vote wade wag wait wake walk wander want warm warn wash waste watch wave wear weave wed weigh welcome whip whirl whisper whistle widen wiggle win wink wipe wish withdraw witness wobble wonder work worry worship wrap wreck wrestle wriggle wring wrinkle write yawn yell yield zip".split(),
    "adj2": "able absent absolute academic accurate active actual adequate adjacent administrative adult advanced afraid aged aggressive agreeable alert alive allergic alone amazing amused ancient angry annoyed anxious apparent appropriate artificial ashamed asleep attractive automatic available average aware awful awkward back bad bald bare basic beautiful beneficial best better bitter blank blind blonde bloody blue blunt bold boring bossy brave brief bright brilliant brisk broken brown brutal busy calm capable careful careless casual cautious certain chaotic charming cheerful childish clean clear clever clumsy coarse cold colorful comfortable common complete complex concerned confident confused conscious consistent constant content cool cooperative correct courageous crazy creative cruel curly curved cute cynical daily damaged dangerous dark dead deaf dear deep defiant delicate delicious delighted democratic dense dependent depressed deserted detailed determined different difficult digital dim diplomatic dirty discreet disgusted dissatisfied distant distinct diverse dizzy dominant doubtful dramatic drastic dread desperate dry dual dull dusty dutiful dynamic eager early easy economic edible educated effective efficient elaborate elastic elderly electric elegant eligible embarrassed emotional empty endless energetic enormous enthusiastic entire equal essential ethical even every evil exact excellent excited exclusive expensive experienced expert explicit exquisite extra extreme fabulous fair faithful false famous fancy fantastic far fascinated fast fat fatal favorite fearful fearless feeble female fertile fierce filthy fine firm fit fixed flaky flat flexible flimsy fluent foamy fond foolish foreign formal fortunate forward foul fragile frank free frequent fresh friendly frightened frozen frustrated full fun fundamental funny furious future fuzzy generous gentle genuine giant gifted glad glamorous global glossy glum gorgeous graceful gracious grand grateful great greedy green grim grimy gross groovy guilty hairy half hallowed handy happy hard harmless harsh healthy heavy helpful helpless hidden high hip hollow holy homeless homely honest honorable hopeless horrible hostile hot huge humble hungry hurt hushed husky icy ideal illegal imaginary immediate immense impolite important impossible impressed impulsive inborn incredible independent individual industrial inevitable informal inner innocent insane instant intelligent intense interactive internal international invisible irritable isolated itchy jealous joyful junior just keen kind knowledgeable known last late latest lazy leading lean legal light likely limited liquid lively local lonely long loose loud lovely lucky lumpy mad magical main major male many married marvelous massive mature mean measly mechanical medium meek melancholy melodic memorable messy mighty mild modern modest moist monthly moral muddy multiple muscular musical mute mysterious narrow nasty native natural naughty nearby neat necessary negative nervous new nice noisy normal notable numerous nutritious obedient obese obvious occasional odd offbeat offensive official oily old only open optimistic orderly ordinary original other outgoing outlandish oval overall overdue overjoyed pale parallel partial particular passive past patient peaceful peculiar perfect permanent personal persuasive petty physical pink plain planned plastic pleasant pleased polite political poor popular portable positive possible powerful practical precious precise pregnant present pretty previous primary prime private probable productive professional profitable profound progressive proud public punctual pure purple quick quiet radical rapid rare rational raw real realistic reasonable recent reckless red redundant regular relevant reliable relieved religious reluctant remarkable remote responsible restless rich ridiculous right ripe risky robust romantic rough round royal rude rural sacred sad safe salty same sandy sane satisfactory scarce scary scientific scrawny secret secure selfish senior sensitive serious sharp shallow shabby shiny short shy sick significant silent silly silver similar simple sincere single skillful skinny sleepy slight slim slippery slow small smart smelly smooth smug soaked social soft solid sophisticated sore sorry sound sour spare sparkly sparse specific spectacular speedy spicy spiritual splendid spoiled spotless square squeaky stable stale standard stark steady steep sticky stiff still stingy stormy straight strange strict strong stubborn stuck stunning sturdy stylish subtle successful sudden sufficient suitable sunny super superb sure surprised suspicious sweet swift symbolic sympathetic talented talkative tall tame tan tasty technical tedious teenage temporary tender tense terrible terrific thick thin thirsty thoughtful tight tiny tired tough toxic tragic tranquil transparent trendy tricky tropical true trustworthy typical ugly ultimate unable unconscious unhealthy unique united unpleasant unsteady upset useful useless vague valid valuable vast venomous verbal very vibrant victorious violent virtual visible vital vivid vocal warm weak wealthy weary weird welcome well-known wet wicked wide wild willing wise witty wonderful wooden worried worthy wrong young yummy".split(),
    "adv": "about above across after again almost already always anywhere around away back backwards barely before behind below beneath besides between beyond carefully certainly clearly close closely completely constantly currently daily deep deeply definitely directly early easily enough entirely equally especially eventually exactly fairly far fast finally firmly first forever forward frequently fully generally gently gladly gradually happily hard hardly here honestly hourly how however immediately instantly instead just late later likely loudly mostly much nearly never next nightly normally not now nowhere often once only otherwise outdoors outright overseas particularly perhaps personally plainly possibly probably quickly quietly quite rarely rather really recently reluctantly repeatedly roughly sadly seldom seriously sharply shortly silently simply slowly smoothly so sometimes somewhere soon speedily steadily still strangely strongly suddenly surely sweetly swiftly then there thoroughly though through throughout tightly today together tomorrow tonight too truly twice unfortunately upwards usually very warmly weekly well whenever wherever why widely wisely yesterday yet".split(),
    "abstract": "ability action activity advantage advice affair agreement aim answer appeal approach argument attention attitude authority background balance belief benefit bit blame blood body breath business calm case cause chance change chaos character choice circumstance claim comfort command comment communication community competition complaint complexity concern condition confidence conflict connection consequence consideration contact content context contract control conversation courage course crime crisis culture curiosity custom danger data decision degree demand democracy desire detail determination development difference difficulty dignity direction disaster discipline discovery discussion disease distance distribution doubt dream duty economy education effect effort emotion emphasis energy enterprise environment error essence event evidence example exception exchange excitement existence experience experiment explanation expression fact failure faith fame fault fear feeling fiction figure form freedom friendship function future gift goal good government growth habit happiness health heart history home honor hope humor horror idea identity illness imagination impact importance impression improvement incident income independence industry influence information injury insight inspiration instinct institution intelligence interest invention issue judgment justice knowledge labor language law leader leadership learning level life lifestyle light link logic loss luck machine management manner mark material matter meaning measure media memory mind minority mistake moment mood morality motive movement mystery nature need network news nonsense note notion object objective obligation observation opinion opportunity option organization outcome pace pain part party past patience peace perception performance period permission person personality perspective philosophy plan pleasure policy politics position possession possibility power practice presence pressure price principle priority problem procedure process profession progress project promise property proportion protection protest purpose quality question rate reaction reality reason recognition record reflection reform relation relationship religion remark reply report reputation request requirement research respect response responsibility result revolution reward risk role routine rule safety satisfaction scale science security sense service shame share shock significance silence similarity situation skill society solution source space spirit standard statement status story strategy stress structure struggle style subject substance success suggestion support surprise system talent task technique technology temper tendency theory thought threat time tolerance tone topic tradition training transition treatment trend trouble trust truth understanding union unity value variety view virtue vision voice war warning way wealth welfare will wisdom wish wonder work worry worth".split(),
    "professions": "actor actress architect artist astronaut athlete author baker banker barber biologist blacksmith builder butcher carpenter cashier chemist chef coach dentist designer detective director economist editor electrician engineer farmer filmmaker firefighter fisherman florist gardener geologist guide hairdresser inspector instructor interpreter journalist judge lawyer librarian lifeguard manager mechanic medic musician nurse optician painter pharmacist photographer physicist pilot planner plumber poet police politician postman professor programmer psychologist publisher receptionist reporter researcher sailor scientist sculptor secretary shoemaker singer soldier surgeon tailor teacher technician translator truck driver veterinarian waiter waitress welder writer".split(),
    "sports": "ball baseball basketball cricket football golf handball hockey rugby soccer tennis volleyball badminton boxing cycling diving fencing gymnastics judo karate rowing running sailing shooting skating skiing surfing swimming table tennis wrestling archery athletics marathon race trophy medal team player coach referee stadium court field track pool gym".split(),
    "music": "music song singer band orchestra choir concert melody rhythm beat note scale tune instrument guitar piano violin cello flute trumpet saxophone drum microphone amplifier stage festival".split(),
    "buildings": "building house apartment cottage cabin castle palace temple church mosque cathedral tower skyscraper factory warehouse barn shed garage greenhouse office library museum gallery theater cinema stadium school university hospital clinic pharmacy bakery butchery shop grocery market mall supermarket store boutique salon barber shop restaurant cafe pub hotel motel hostel airport station terminal dock harbor".split(),
    "vehicles": "car van truck lorry taxi bus coach minibus tram train subway metro light rail bicycle motorcycle scooter tricycle skateboard roller skates wheelchair ambulance fire engine police car crane bulldozer excavator tractor combine harvester forklift ship boat ferry yacht submarine hovercraft airplane helicopter jet glider hot air balloon rocket spaceship".split(),
    "materials": "wood metal iron steel copper bronze aluminum tin gold silver platinum glass plastic rubber leather wool cotton silk linen denim fur paper cardboard ceramics pottery brick stone marble granite slate concrete cement plaster mortar tar asphalt clay sand gravel pebble crystal gemstone diamond pearl ivory coral shell feather bone horn antler".split(),
    "shapes": "circle square triangle rectangle oval diamond heart star crescent cross arrow line curve angle edge corner flat round long short thick thin narrow wide hollow solid".split(),
    "direction": "up down left right forward backward sideways north south east west center middle top bottom front back inside outside near far here there".split(),
    "time2": "january february march april may june july august september october november december monday tuesday wednesday thursday friday saturday sunday morning noon afternoon evening dusk night midnight dawn sunrise sunset weekend weekday holiday vacation".split(),
    "health": "health healthy sick ill disease illness fever cold flu cough sneeze headache stomachache toothache injury wound scar bruise burn cut scratch blood bone muscle joint pain ache medicine pill tablet syrup injection vaccine doctor nurse hospital clinic pharmacy dentist checkup exercise diet rest sleep energy".split(),
    "emotions": "love joy happiness sadness anger fear surprise disgust hope pride shame guilt envy jealousy compassion gratitude sympathy pity regret relief loneliness nostalgia excitement boredom anxiety stress calm peace worry doubt confusion trust respect admiration awe wonder curiosity interest passion delight thrill satisfaction contentment despair grief sorrow rage frustration annoyance irritation resentment forgiveness patience kindness gentleness courage bravery confidence humility modesty honesty sincerity loyalty".split(),
    "money": "money cash coin banknote dollar cent euro pound yen yuan currency price cost value bargain discount sale receipt invoice budget income salary wage tax loan debt interest savings wealth poverty charity donation invest investment business company profit loss credit debit payment refund".split(),
    "geo": "earth world globe map atlas ocean sea river lake mountain hill valley desert island continent country nation state province city town village capital region territory border coast shore beach forest jungle savanna tundra glacier ice cap volcano canyon cliff plateau plain prairie steppe".split(),
    "space": "space universe galaxy star planet earth moon sun asteroid comet meteor meteorite orbit eclipse gravity telescope astronaut rocket satellite spaceship station".split(),
    "measure": "meter centimeter kilometer inch foot yard mile gram kilogram pound ounce liter gallon degree percent fraction number amount size length width height depth weight volume speed distance temperature".split(),
    "school2": "teacher student pupil classroom desk board chalk eraser homework exercise lesson subject mathematics science history geography english biology chemistry physics art music physical education test exam grade score result certificate graduation school college university".split(),
    "home2": "apartment house room kitchen bedroom bathroom living room dining room hallway balcony garden yard garage basement attic stairs elevator doorbell kitchen sink faucet toilet shower bathtub mirror cabinet cupboard shelf drawer wardrobe closet mattress pillow blanket sheet towel".split(),
    "tech2": "app application software program website email message chat video call screen keyboard mouse speaker camera printer scanner router modem battery charger cable wire plug socket switch remote control drone robot sensor chip processor memory storage cloud network password account username".split(),
    "travel2": "journey trip voyage tour travel visit destination route itinerary departure arrival ticket luggage baggage passport visa currency language culture food souvenir photograph experience adventure exploration".split(),
}

for cat, words in EXTRA.items():
    WORDS[cat] = words

# dedupe: keep first category occurrence, drop repeats
seen = set()
for cat in list(WORDS.keys()):
    keep = []
    for w in WORDS[cat]:
        if w not in seen:
            seen.add(w)
            keep.append(w)
        else:
            print(f"  dup {w} (in {cat}) -> dropped")
    WORDS[cat] = keep

ALL_WORDS = [w for cat in WORDS.values() for w in cat]
print(f"total words: {len(ALL_WORDS)}")

# word -> column
CATEGORY_MAP = {w: cat for cat, ws in WORDS.items() for w in ws}

# ---- relationship pairs (word A is commonly connected to word B) ----
PAIRS = {
    "dog": ["bone", "bark", "tail", "pet"], "cat": ["milk", "mouse", "sleep"],
    "bird": ["fly", "sky", "song"], "fish": ["water", "swim"],
    "cow": ["milk", "grass"], "horse": ["ride", "farm"], "sheep": ["wool", "grass"],
    "apple": ["red", "tree", "eat"], "banana": ["yellow", "peel"],
    "bread": ["butter", "eat"], "egg": ["chicken", "breakfast"],
    "milk": ["white", "drink", "cow"], "water": ["drink", "clean", "river"],
    "sun": ["hot", "bright", "moon"], "moon": ["night", "star"],
    "rain": ["cloud", "wet", "umbrella"], "snow": ["cold", "white"],
    "hand": ["finger", "wash"], "eye": ["see", "blue"], "ear": ["hear"],
    "nose": ["smell"], "mouth": ["eat", "talk"], "foot": ["shoe", "walk"],
    "bed": ["sleep", "soft"], "door": ["open", "close", "room"],
    "window": ["glass", "look"], "table": ["chair", "eat"],
    "book": ["read", "page", "school"], "pen": ["write", "paper"],
    "car": ["drive", "road", "fast"], "bus": ["ride", "school"],
    "train": ["station", "fast"], "plane": ["fly", "sky", "airport"],
    "boat": ["water", "river"], "phone": ["call", "message"],
    "computer": ["keyboard", "screen", "work"], "ball": ["play", "kick", "throw"],
    "tree": ["leaf", "grow", "forest"], "flower": ["grow", "smell", "pretty"],
    "star": ["night", "sky", "shiny"], "rainbow": ["color", "sky", "after rain"],
    "music": ["song", "listen", "dance"], "game": ["play", "fun"],
}

PAIR_WORDS = {w: PAIRS.get(w, []) for w in ALL_WORDS}

# ---- sentence templates ----
TEMPLATES = {
    "people": [
        "the {w} is {adj}.", "i see the {w}.", "the {w} and i are {adj}.",
        "my {w} is {adj}.", "the {w} loves {food}.",
    ],
    "body": [
        "my {w} is {adj}.", "i touch my {w}.", "the {w} can {action}.",
        "i wash my {w}.", "my {w} and my {w2} are {adj}.",
    ],
    "animals": [
        "the {w} is {adj}.", "the {w} can {action}.", "i see the {w} in the {place}.",
        "the {w} eats {food}.", "the {w} and the {animal2} are {adj}.",
    ],
    "food": [
        "i eat {w}.", "the {w} is {adj}.", "i like {w} and {food2}.",
        "{w} is good for you.", "mother cooks {w} in the kitchen.",
    ],
    "home": [
        "the {w} is in the {place}.", "my {w} is {adj}.", "i clean the {w}.",
        "the {w} is in my house.", "the {w} and the {obj2} are {adj}.",
    ],
    "objects": [
        "the {w} is {adj}.", "i have a {w}.", "i use the {w} every day.",
        "my {w} is on the table.", "the {w} is in the {place}.",
    ],
    "nature": [
        "the {w} is {adj}.", "i see the {w} in the sky.", "the {w} is outside.",
        "we look at the {w}.", "the {w} and the {nat2} are {adj}.",
    ],
    "weather": [
        "today is {w}.", "the weather is {w} today.", "i like {w} days.",
        "when it is {w} i stay home.", "it is {w} in the morning.",
    ],
    "time": [
        "i wake up in the {w}.", "the {w} is {adj}.", "we play in the {w}.",
        "my birthday is in the {w}.", "i sleep at {w}.",
    ],
    "numbers": [
        "i have {w} {obj2}s.", "count to {w}.", "the {w} dog is {adj}.",
        "i see {w} {obj2}s.", "we have {w} {food2}s.",
    ],
    "colors": [
        "the {obj2} is {w}.", "i like {w}.", "my {obj2} is {w}.",
        "the {w} {obj2} is {adj}.", "i paint with {w}.",
    ],
    "clothes": [
        "my {w} is {adj}.", "i wear a {w}.", "the {w} is clean.",
        "i put on my {w}.", "my {w} is in the {place}.",
    ],
    "school": [
        "i {w} at school.", "we {w} every day.", "i {w} with my teacher.",
        "i {w} in the morning.", "we {w} and {w2} in class.",
    ],
    "actions": [
        "i {w} now.", "we {w} together.", "i {w} every morning.",
        "i want to {w}.", "the {obj2} makes me {w}.",
    ],
    "feelings": [
        "i am {w}.", "the {obj2} is {w}.", "i feel {w} today.",
        "my {food2} is {w}.", "the {place} is {w}.",
    ],
    "places": [
        "i go to the {w}.", "the {w} is {adj}.", "we visit the {w} on Sunday.",
        "the {w} is near my house.", "i like the {w}.",
    ],
    "travel": [
        "i {w} to the {place}.", "we {w} together.", "i {w} in the summer.",
        "my family {w} on holiday.", "i {w} home.",
    ],
    "family": [
        "we {w} together.", "i {w} with my family.", "we {w} at home.",
        "i {w} every day.", "we {w} and have fun.",
    ],
    "technology": [
        "i use the {w} every day.", "my {w} is {adj}.", "the {w} works well.",
        "i look at my {w}.", "the {w} is new.",
    ],
    "grammar": [
        "{w} is the best.", "{w} like apples.", "{w} go to school.",
        "{w} are my friends.", "{w} is my dog.",
    ],
    "prepositions": [
        "the ball is {w} the box.", "i sit {w} my mother.",
        "the book is {w} the table.", "we walk {w} the park.",
        "i put the toy {w} the bed.",
    ],
}

TEMPLATE_ALIAS = {
    "animals2": "animals", "food2": "food", "objects2": "objects",
    "nature2": "nature", "verbs2": "actions", "adj2": "feelings",
    "adv": "prepositions", "abstract": "objects", "professions": "people",
    "sports": "actions", "music": "objects", "buildings": "places",
    "vehicles": "objects", "materials": "objects", "shapes": "objects",
    "direction": "prepositions", "time2": "time", "health": "body",
    "emotions": "feelings", "money": "objects", "geo": "places",
    "space": "nature", "measure": "numbers", "school2": "school",
    "home2": "home", "tech2": "technology", "travel2": "travel",
}

ADJ = "good nice big small fast slow happy sad new old hot cold soft hard clean bright".split()
ACTION = "run walk jump play eat drink read write see hear sing dance".split()
FOODS = "apple banana bread egg milk rice soup cake cheese water juice orange meat".split()
PLACES = "park house school garden river beach shop farm kitchen room forest".split()
OBJECTS = "ball book pen car bus train phone box bag cup chair table lamp".split()
ANIMALS = "dog cat bird fish horse cow sheep chicken duck rabbit bear lion".split()
NATURE = "sun moon star cloud rain river tree flower".split()


def pick(pool):
    return rng.choice(pool)


def make_sentence(word, cat):
    tcat = TEMPLATE_ALIAS.get(cat, cat)
    t = rng.choice(TEMPLATES[tcat])
    subs = {
        "{w}": word,
        "{adj}": pick(ADJ),
        "{action}": pick(ACTION),
        "{food}": pick(FOODS),
        "{food2}": pick(FOODS),
        "{place}": pick(PLACES),
        "{obj2}": pick(OBJECTS),
        "{animal2}": pick(ANIMALS),
        "{nat2}": pick(NATURE),
        "{w2}": word,
    }
    out = t
    for k, v in subs.items():
        out = out.replace(k, v)
    # inject relationship pairs when available
    pairs = PAIR_WORDS.get(word, [])
    if pairs and rng.random() < 0.5:
        p = rng.choice(pairs)
        out = out.replace(word, f"{word} and {p}", 1) if rng.random() < 0.3 else out
        out = f"the {word} is near the {p}." if rng.random() < 0.2 else out
    return out.capitalize()


def main():
    sentences = []
    for cat, words in WORDS.items():
        for w in words:
            n = rng.randint(3, 5)
            for _ in range(n):
                sentences.append(make_sentence(w, cat))
    rng.shuffle(sentences)

    (CUR / "words_2000.txt").write_text("\n".join(ALL_WORDS))
    (CUR / "sentences.txt").write_text("\n".join(sentences))
    (CUR / "categories.json").write_text(json.dumps(CATEGORY_MAP, indent=0))
    chars = sum(len(s) + 1 for s in sentences)
    print(f"sentences: {len(sentences)} | chars: {chars}")
    print("sample:")
    for s in sentences[:8]:
        print("  ", s)


if __name__ == "__main__":
    main()