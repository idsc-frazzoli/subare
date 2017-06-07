// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import java.util.List;
import java.util.Random;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.color.Colorscheme;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.opt.Interpolation;

enum CarRentalHelper {
  ;
  private static final Tensor BASE = Tensors.vector(255);

  public static Tensor render(CarRental carRental, DiscreteVs vs) {
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros(21, 21, 4);
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).flatten(0));
    for (Tensor state : carRental.states()) {
      Scalar sca = scaled.value(state);
      int x = state.Get(0).number().intValue();
      int y = state.Get(1).number().intValue();
      tensor.set(colorscheme.get(BASE.multiply(sca)), x, y);
    }
    return ImageResize.of(tensor, 4);
  }

  public static Tensor render(CarRental gambler, PolicyInterface policyInterface) {
    Interpolation colorscheme = Colorscheme.classic();
    final Tensor tensor = Array.zeros(21, 21, 4);
    PolicyWrap policyWrap = new PolicyWrap(policyInterface, new Random());
    for (Tensor state : gambler.states()) {
      Tensor action = policyWrap.next(state, gambler.actions(state));
      int x = state.Get(0).number().intValue();
      int y = state.Get(1).number().intValue();
      Scalar sca = action.Get().add(RealScalar.of(5)).divide(RealScalar.of(10));
      tensor.set(colorscheme.get(BASE.multiply(sca)), x, y);
    }
    return ImageResize.of(tensor, 4);
  }

  public static Tensor joinAll(CarRental gambler, DiscreteVs vs) {
    Tensor im1 = render(gambler, vs);
    PolicyInterface pi = GreedyPolicy.bestEquiprobable(gambler, vs);
    Tensor im2 = render(gambler, pi);
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 4 * 2);
    return Join.of(0, im1, Array.zeros(list), im2);
  }
}
