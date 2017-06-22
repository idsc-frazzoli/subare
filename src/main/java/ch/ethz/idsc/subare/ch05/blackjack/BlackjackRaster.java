// code by jph
package ch.ethz.idsc.subare.ch05.blackjack;

import java.awt.Point;
import java.util.Arrays;

import ch.ethz.idsc.subare.core.util.gfx.DefaultStateRaster;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

class BlackjackRaster extends DefaultStateRaster {
  public BlackjackRaster(Blackjack blackjack) {
    super(blackjack, Arrays.asList(20 + 2, 10));
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

  @Override
  public Scalar scaleLoss() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Scalar scaleQdelta() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public int joinAlongDimension() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public int magify() {
    return 5;
  }
}
