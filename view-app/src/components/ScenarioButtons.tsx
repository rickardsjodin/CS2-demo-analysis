import './ScenarioButtons.css';

interface ScenarioButtonsProps {
  onScenarioLoad: (scenarioData: { [key: string]: number | string }) => void;
  disabled: boolean;
}

const scenarios = {
  default: {
    round_time_left: 75,
    bomb_time_left: 40,
    cts_alive: 5,
    ts_alive: 5,
    bomb_planted: 0,
    hp_ct: 500,
    hp_t: 500,
    ct_main_weapons: 3,
    t_main_weapons: 3,
    defusers: 2,
    ct_armor: 4,
    t_armor: 4,
    ct_avg_gear: 'tier1_rifle',
    t_avg_gear: 'tier1_rifle',
  },
  'ct-advantage': {
    round_time_left: 60,
    cts_alive: 5,
    ts_alive: 3,
    bomb_planted: 0,
    hp_ct: 450,
    hp_t: 200,
    ct_main_weapons: 5,
    t_main_weapons: 2,
    defusers: 3,
    ct_avg_gear: 'tier1_rifle',
    t_avg_gear: 'smg_shotgun',
  },
  't-advantage': {
    round_time_left: 45,
    cts_alive: 2,
    ts_alive: 4,
    bomb_planted: 0,
    hp_ct: 150,
    hp_t: 350,
    ct_main_weapons: 2,
    t_main_weapons: 4,
    defusers: 1,
    ct_avg_gear: 'upgraded_pistol',
    t_avg_gear: 'tier1_rifle',
  },
  'bomb-planted': {
    round_time_left: 30,
    bomb_time_left: 30,
    cts_alive: 3,
    ts_alive: 2,
    bomb_planted: 1,
    hp_ct: 250,
    hp_t: 150,
    defusers: 2,
    ct_avg_gear: 'tier1_rifle',
    t_avg_gear: 'tier1_rifle',
  },
  retake: {
    round_time_left: 20,
    bomb_time_left: 20,
    cts_alive: 4,
    ts_alive: 1,
    bomb_planted: 1,
    hp_ct: 300,
    hp_t: 80,
    defusers: 3,
    ct_avg_gear: 'tier1_rifle',
    t_avg_gear: 'upgraded_pistol',
  },
};

export default function ScenarioButtons({
  onScenarioLoad,
  disabled,
}: ScenarioButtonsProps) {
  const handleScenarioClick = (scenarioKey: string) => {
    if (disabled) return;

    const scenarioData = scenarios[scenarioKey as keyof typeof scenarios];
    if (scenarioData) {
      onScenarioLoad(scenarioData);
    }
  };

  return (
    <section className='scenarios-section'>
      <h2>⚡ Quick Scenarios</h2>
      <div className='scenario-buttons'>
        <button
          className='scenario-btn'
          onClick={() => handleScenarioClick('default')}
          disabled={disabled}
        >
          Balanced Round
        </button>
        <button
          className='scenario-btn'
          onClick={() => handleScenarioClick('ct-advantage')}
          disabled={disabled}
        >
          CT Advantage
        </button>
        <button
          className='scenario-btn'
          onClick={() => handleScenarioClick('t-advantage')}
          disabled={disabled}
        >
          T Advantage
        </button>
        <button
          className='scenario-btn'
          onClick={() => handleScenarioClick('bomb-planted')}
          disabled={disabled}
        >
          Bomb Planted
        </button>
        <button
          className='scenario-btn'
          onClick={() => handleScenarioClick('retake')}
          disabled={disabled}
        >
          Retake Situation
        </button>
      </div>
    </section>
  );
}
