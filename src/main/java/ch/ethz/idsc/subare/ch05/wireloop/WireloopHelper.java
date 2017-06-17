// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.io.File;
import java.util.List;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.subare.core.util.StateRasters;
import ch.ethz.idsc.subare.core.util.TensorValuesUtils;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.subare.util.RobustArgMax;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.io.Import;
import ch.ethz.idsc.tensor.sca.Clip;

enum WireloopHelper {
  ;
  private static final int MAGNIFY = 6; // 3

  static Wireloop create(String trackName, Function<Tensor, Scalar> function, Function<Tensor, Scalar> stepCost) throws Exception {
    String path = "".getClass().getResource("/ch05/" + trackName + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Wireloop(image, function, stepCost);
  }

  static Wireloop create(String trackName, Function<Tensor, Scalar> function) throws Exception {
    return create(trackName, function, action -> RealScalar.ZERO);
  }

  static DiscreteQsa getOptimalQsa(Wireloop wireloop) {
    return ActionValueIterations.solve(wireloop, DecimalScalar.of(.0001));
  }

  public static Tensor render(Wireloop wireloop, DiscreteQsa qsa) {
    return render(wireloop, DiscreteUtils.createVs(wireloop, qsa));
  }

  public static Tensor renderActions(Wireloop wireloop, QsaInterface qsa) {
    DiscreteVs vs = DiscreteVs.build(wireloop);
    for (Tensor state : wireloop.startStates()) {
      Tensor tensor = Tensor.of(wireloop.actions(state).flatten(0).map(action -> qsa.value(state, action)));
      int index;
      // index = ArgMax.of(tensor);
      index = RobustArgMax.of(tensor);
      vs.assign(state, RealScalar.of(index * 0.25 + 0.185));
    }
    return render_asIs(wireloop, vs);
  }

  public static Tensor render(Wireloop wireloop, DiscreteQsa ref, DiscreteQsa qsa) {
    Tensor im1 = render(wireloop, DiscreteUtils.createVs(wireloop, qsa));
    DiscreteVs loss = Loss.perState(wireloop, ref, qsa);
    loss = loss.create(loss.values().flatten(0) //
        .map(tensor -> tensor.multiply(RealScalar.of(100))) //
        .map(Clip.UNIT::of));
    Tensor im2 = render_asIs(wireloop, loss);
    Tensor im3 = renderActions(wireloop, qsa);
    List<Integer> dimensions = Dimensions.of(im1);
    dimensions.set(0, MAGNIFY);
    return Join.of(im1, Array.zeros(dimensions), im2, Array.zeros(dimensions), im3);
  }

  public static Tensor render(Wireloop wireloop, DiscreteVs vs) {
    return render_asIs(wireloop, TensorValuesUtils.rescaled(vs));
  }

  public static Tensor render_asIs(Wireloop wireloop, DiscreteVs vs) {
    return ImageResize.of(StateRasters.render( //
        new WireloopRaster(wireloop), vs), MAGNIFY);
  }

  /***************************************************/
  static Scalar id_x(Tensor state) {
    return state.Get(0);
  }
}
