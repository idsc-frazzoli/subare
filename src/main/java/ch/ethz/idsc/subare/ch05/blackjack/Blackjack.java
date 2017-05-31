// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;

/** Example 5.1 p.101 */
class Blackjack implements MonteCarloInterface, EpisodeSupplier {
  static final Tensor END_WIN = Tensors.vector(1);
  static final Tensor END_DRAW = Tensors.vector(0);
  static final Tensor END_LOSS = Tensors.vector(-1);
  // ---
  Random random = new Random();
  // states are product of
  // dealer showing: 1(=A), 2, 3, 4, 5, 6, 7, 8, 9, {T, J, Q, K} - #=10
  // player sum 12, 13, ..., 21 - #=10
  // player has usable ace {0, 1}
  Tensor states = Tensors.empty();
  // Tensor startStates = null;
  Tensor actions = Tensors.vector(0, 1); // stay, hit

  public Blackjack() {
    for (int ace = 0; ace < 2; ++ace)
      for (int player = 12; player <= 21; ++player)
        for (int dealer = 1; dealer <= 10; ++dealer)
          states.append(Tensors.vector(ace, player, dealer));
    // startStates = states.copy();
    // ---
    states.append(END_WIN);
    states.append(END_DRAW);
    states.append(END_LOSS);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return actions;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    // player stays, the next state is terminal
    if (action.equals(ZeroScalar.get())) { // stay
      // TODO natural: ACE+10-card
      int dealer = state.Get(2).number().intValue();
      boolean usableAce = dealer == 1;
      if (usableAce)
        dealer += 10;
      while (dealer < 17) {
        int index = random.nextInt(PlayingCard.values().length);
        PlayingCard playingCard = PlayingCard.values()[index];
        dealer += playingCard.value;
        if (21 < dealer && usableAce) {
          dealer -= 10;
          usableAce = false;
        }
      }
      if (dealer <= 21) {
        int player = state.Get(1).number().intValue();
        if (player < dealer)
          return END_LOSS; // dealer is closer to 21
        if (player > dealer)
          return END_WIN; // player is closer to 21
        return END_DRAW; // draw
      }
      return END_WIN; // dealer goes bust
    }
    // player hits
    int player = state.Get(1).number().intValue();
    int index = random.nextInt(PlayingCard.values().length);
    PlayingCard playingCard = PlayingCard.values()[index];
    player += playingCard.value;
    if (player <= 21) {
      Tensor next = state.copy();
      next.set(RealScalar.of(player), 1);
      return next;
    }
    if (state.Get(0).number().intValue() == 0) // has usable ace
      return END_LOSS;
    player -= 10;
    Tensor next = state.copy();
    next.set(ZeroScalar.get(), 0);
    next.set(RealScalar.of(player), 1);
    return next;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (!isTerminal(state) && isTerminal(next))
      return next.Get(0);
    return ZeroScalar.get();
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return state.length() == 1;
  }

  @Override
  public EpisodeInterface kickoff(PolicyInterface policyInterface) {
    Tensor start = states.get(random.nextInt(states.length() - 3));
    if (isTerminal(start))
      throw new RuntimeException();
    return new MonteCarloEpisode(this, policyInterface, start);
  }
}