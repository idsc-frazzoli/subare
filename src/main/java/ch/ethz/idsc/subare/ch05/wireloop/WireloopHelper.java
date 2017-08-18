// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import java.io.File;
import java.util.List;
import java.util.function.Function;

import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.core.util.Loss;
import ch.ethz.idsc.subare.core.util.gfx.StateRasters;
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
  static Wireloop create(String trackName, Function<Tensor, Scalar> function, WireloopReward wireloopReward) throws Exception {
    String path = "".getClass().getResource("/ch05/" + trackName + ".png").getPath();
    Tensor image = Import.of(new File(path)).unmodifiable();
    return new Wireloop(image, function, wireloopReward);
  }

  static Wireloop create(String trackName, Function<Tensor, Scalar> function) throws Exception {
    return create(trackName, function, WireloopReward.freeSteps());
  }

  static DiscreteQsa getOptimalQsa(Wireloop wireloop) {
    return ActionValueIterations.solve(wireloop, DecimalScalar.of(.0001));
  }

  private static Tensor renderActions(Wireloop wireloop, QsaInterface qsa) {
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteVs vs = DiscreteVs.build(wireloop);
    for (Tensor state : wireloop.startStates()) {
      Tensor tensor = Tensor.of(wireloop.actions(state).flatten(0).map(action -> qsa.value(state, action)));
      int index = RobustArgMax.of(tensor);
      vs.assign(state, RealScalar.of(index * 0.25 + 0.185));
    }
    return StateRasters.vs(wireloopRaster, vs);
  }

  public static Tensor render(WireloopRaster wireloopRaster, DiscreteQsa ref, DiscreteQsa qsa) {
    Tensor image1 = StateRasters.vs_rescale(wireloopRaster, qsa);
    DiscreteVs loss = Loss.perState(wireloopRaster.discreteModel(), ref, qsa);
    loss = loss.create(loss.values().flatten(0) //
        .map(tensor -> tensor.multiply(wireloopRaster.scaleLoss())) //
        .map(Clip.unit()::of));
    Tensor image2 = StateRasters.vs(wireloopRaster, loss);
    Tensor image3 = renderActions((Wireloop) wireloopRaster.discreteModel(), qsa);
    List<Integer> dimensions = Dimensions.of(image1);
    dimensions.set(wireloopRaster.joinAlongDimension(), wireloopRaster.magnify());
    return Join.of(image1, Array.zeros(dimensions), image2, Array.zeros(dimensions), image3);
  }
}
