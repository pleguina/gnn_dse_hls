<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" top="graphsage_network_qat" name="graphsage_qat" ideType="classic">
    <files>
        <file name="graphsage_layer_qat.h" sc="0" tb="false" cflags="-I." csimflags="" blackbox="false"/>
        <file name="graphsage_layer_qat.cpp" sc="0" tb="false" cflags="-I." csimflags="" blackbox="false"/>
        <file name="../../testbench_qat.cpp" sc="0" tb="1" cflags="-I../../. -Wno-unknown-pragmas" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

