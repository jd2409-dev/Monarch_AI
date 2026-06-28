"""Quick test of agent with local LLM."""
import sys; sys.path.insert(0, '.')
from soma_mythos_ehra.arc3.interactive_agent import InteractiveAgent, AgentConfig

config = AgentConfig(
    max_steps=80, max_episodes=2, ensemble_size=2,
    latent_dim=128, train_steps_per_episode=15,
    evolve_every_n_episodes=2, use_llm=True, verbose=True,
)
agent = InteractiveAgent(config)
stats = agent.play_game('ls20', max_episodes=2)
print(f'\n=== SUMMARY ===')
for i, s in enumerate(stats):
    print(f'  Ep {i+1}: won={s.won}, steps={s.total_steps}, code_scores={[f"{x:.2f}" for x in s.code_scores]}')
print(f'Buffer: {len(agent.buffer)}')
print(agent.efficiency.get_efficiency_report())
