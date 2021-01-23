// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.awt.Dimension;
import java.awt.Point;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.util.gfx.StateRaster;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
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
  public Dimension dimensionStateRaster() {
    return new Dimension(20 + 2, 20);
  }

  @Override
  public Point point(Tensor state) {
    if (state.length() == 3) {
      int useAce = Scalars.intValueExact(state.Get(0));
      int player = Scalars.intValueExact(state.Get(1)) - 12;
      int dealer = Scalars.intValueExact(state.Get(2)) - 1;
      return new Point(dealer + (10 + 2) * useAce, 9 - player);
    }
    return null;
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.ONE;
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.ONE;
  }

  @Override
  public int joinAlongDimension() {
    return 0;
  }

  @Override
  public int magnify() {
    return 5;
  }
}
