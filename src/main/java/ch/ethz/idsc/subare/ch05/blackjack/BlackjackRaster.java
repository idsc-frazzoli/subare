// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.StateRaster;
import ch.ethz.idsc.tensor.Tensor;

class BlackjackRaster implements StateRaster {
  private final Blackjack blackjack;

  public BlackjackRaster(Blackjack blackjack) {
    this.blackjack = blackjack;
  }

  @Override
  public DiscreteModel discreteModel() {
    return blackjack;
  }

  @Override
  public Dimension dimension() {
    return new Dimension(20 + 2, 10);
  }

  @Override
  public Point point(Tensor state) {
    if (state.length() == 3) {
      int useAce = state.Get(0).number().intValue();
      int player = state.Get(1).number().intValue() - 12;
      int dealer = state.Get(2).number().intValue() - 1;
      return new Point(dealer + (10 + 2) * useAce, 9 - player);
    }
    return null;
  }
}
