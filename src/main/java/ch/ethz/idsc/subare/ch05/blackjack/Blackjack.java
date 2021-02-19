// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.util.Random;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.sca.Sign;

/** Example 5.1 p.101: Blackjack
 * 
 * Reference: Widrow, Gupta, and Maitra (1973) */
class Blackjack implements MonteCarloInterface {
  private static final Tensor END_WIN = Tensors.vector(1);
  private static final Tensor END_DRAW = Tensors.vector(0);
  private static final Tensor END_LOSS = Tensors.vector(-1);
  // ---
  private Random random = new Random();
  // states are product of
  // dealer showing: 1(=A), 2, 3, 4, 5, 6, 7, 8, 9, {T, J, Q, K} - #=10
  // player sum 12, 13, ..., 21 - #=10
  // player has usable ace {0, 1}
  private final Tensor states = Tensors.empty();
  private final Tensor startStates;
  private final Tensor actions = Tensors.vector(0, 1); // stay, or hit
  private final Tensor actionsTerminal = Tensors.vector(0); // do nothing

  public Blackjack() {
    for (int ace = 0; ace < 2; ++ace)
      for (int player = 12; player <= 21; ++player)
        for (int dealer = 1; dealer <= 10; ++dealer)
          states.append(Tensors.vector(ace, player, dealer));
    startStates = states.copy().unmodifiable();
    // ---
    states.append(END_WIN);
    states.append(END_DRAW);
    states.append(END_LOSS);
  }

  @Override // from DiscreteModel
  public Tensor states() {
    return states;
  }

  @Override // from DiscreteModel
  public Tensor actions(Tensor state) {
    return isTerminal(state) //
        ? actionsTerminal
        : actions;
  }

  @Override // from DiscountInterface
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  /**************************************************/
  @Override // from MoveInterface
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    // player stays, the next state is terminal
    if (Scalars.isZero((Scalar) action)) { // stay
      // TODO natural: ACE+10-card
      int dealer = Scalars.intValueExact(state.Get(2));
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
        return Tensors.of(Sign.FUNCTION.apply(RealScalar.of(player - dealer)));
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
    next.set(RealScalar.ZERO, 0);
    next.set(RealScalar.of(player), 1);
    return next;
  }

  @Override // from RewardInterface
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return !isTerminal(state) && isTerminal(next) //
        ? next.Get(0) // transition into terminal state
        : RealScalar.ZERO;
  }

  /**************************************************/
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return startStates;
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.length() == 1;
  }
}
