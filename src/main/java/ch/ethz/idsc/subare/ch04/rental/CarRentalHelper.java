// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import java.util.List;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Rescale;
import ch.ethz.idsc.tensor.img.ArrayPlot;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.img.ImageResize;

enum CarRentalHelper {
  ;
  public static Tensor render(CarRental carRental, DiscreteVs vs) {
    // TODO use createRaster
    final Tensor tensor = Array.of(list -> DoubleScalar.INDETERMINATE, 21, 21);
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).flatten(0));
    for (Tensor state : carRental.states()) {
      Scalar sca = scaled.value(state);
      int x = state.Get(0).number().intValue();
      int y = state.Get(1).number().intValue();
      tensor.set(sca, x, y);
    }
    Tensor image = ArrayPlot.of(tensor, ColorDataGradients.CLASSIC);
    return ImageResize.nearest(image, 4);
  }

  public static Tensor render(CarRental carRental, Policy policy) {
    final Tensor tensor = Array.of(list -> DoubleScalar.INDETERMINATE, 21, 21);
    PolicyWrap policyWrap = new PolicyWrap(policy);
    for (Tensor state : carRental.states()) {
      Tensor action = policyWrap.next(state, carRental.actions(state));
      int x = state.Get(0).number().intValue();
      int y = state.Get(1).number().intValue();
      Scalar sca = action.Get().add(RealScalar.of(5)).divide(RealScalar.of(10));
      tensor.set(sca, x, y);
    }
    Tensor image = ArrayPlot.of(tensor, ColorDataGradients.CLASSIC);
    return ImageResize.nearest(image, 4);
  }

  public static Tensor joinAll(CarRental carRental, DiscreteVs vs) {
    Tensor im1 = render(carRental, vs);
    Policy pi = GreedyPolicy.bestEquiprobable(carRental, vs);
    Tensor im2 = render(carRental, pi);
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 4 * 2);
    return Join.of(0, im1, Array.zeros(list), im2);
  }
}
